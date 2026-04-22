"""
Memento-Skills Demo  (Ollama edition)
======================================
A self-contained Python demo that illustrates the two core mechanisms of
https://github.com/Memento-Teams/Memento-Skills

  Example 1 — Skill Routing
    Given a user query the agent scores its skill library with BM25,
    picks the best skill, builds a prompt from SKILL.md, calls Ollama,
    and returns the answer.

  Example 2 — Read → Execute → Reflect → Write (Self-Evolution Loop)
    The agent attempts a task with a deliberately weak skill.  Execution
    fails (the LLM is told the instructions are broken and asked to flag
    it).  The Reflector rewrites SKILL.md, the skill is persisted back
    into the library, and the second attempt succeeds with correct output.

Usage:
    pip install requests rich          # one-time
    python memento_demo.py             # default: qwen3.5:9b
    python memento_demo.py --model qwen3:30b
    python memento_demo.py --model nemotron-cascade-2:latest
    python memento_demo.py --url http://192.168.1.10:11434   # remote Ollama

Available on your machine:
    nemotron-cascade-2:latest  (31.6 B Q4_K_M)
    qwen3.5:35b                (36.0 B Q4_K_M)
    qwen3.5:9b                 (9.7 B  Q4_K_M)   ← default
    mdq100/qwen3.5-coder:35b   (36.0 B Q4_K_M)
    qwen3:30b                  (30.5 B Q4_K_M)
    gpt-oss:120b               (116.8 B MXFP4)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from dataclasses import dataclass
from enum import Enum, auto

import requests

# ── pretty output (optional) ────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.spinner import Spinner
    from rich.live import Live
    from rich import box

    RICH = True
    console = Console()
except ImportError:
    RICH = False

    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60}  {t}")

    console = _FallbackConsole()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_URL   = "http://localhost:11434"

# ════════════════════════════════════════════════════════════════════════════
# DOMAIN MODELS  (mirrors core/skill/schema.py)
# ════════════════════════════════════════════════════════════════════════════

class ExecutionMode(Enum):
    KNOWLEDGE = auto()   # pure reasoning — LLM answers from skill instructions
    PLAYBOOK  = auto()   # has runnable tool calls / scripts


class SkillStatus(Enum):
    SUCCESS = auto()
    FAILED  = auto()


@dataclass
class Skill:
    """
    Atomic capability unit.  In the real system every skill lives in its own
    folder and is described by a SKILL.md file fed verbatim to the LLM.
    """
    name: str
    description: str
    content: str            # the SKILL.md text
    execution_mode: ExecutionMode = ExecutionMode.KNOWLEDGE
    utility_score: float = 1.0
    version: int = 1


@dataclass
class SkillExecutionOutcome:
    success: bool
    result: str
    error: str | None = None
    skill_name: str = ""


# ════════════════════════════════════════════════════════════════════════════
# REAL OLLAMA LLM CLIENT  (replaces StubLLM)
# ════════════════════════════════════════════════════════════════════════════

class OllamaLLM:
    """
    Thin wrapper around the Ollama /api/chat REST endpoint.

    The real Memento-Skills system uses litellm with the profile format
    ``ollama/<model>``.  We call the Ollama API directly so the demo has
    zero extra dependencies beyond `requests`.

    Parameters
    ----------
    model : str
        Any model name visible in ``ollama list`` on your machine.
    base_url : str
        Ollama server URL, e.g. ``http://localhost:11434``.
    timeout : int
        Per-request timeout in seconds.  Large models can be slow on first
        load; 300 s is a safe default.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_URL,
                 timeout: int = 300) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._endpoint = f"{self.base_url}/api/chat"

    # ── health check ──────────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def available_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── core call ─────────────────────────────────────────────────────────
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request to Ollama and return the assistant text.

        Handles both standard models and "thinking" models (Qwen3, QwQ, …)
        that emit a <think> block before the final answer.  When content is
        empty but the thinking field is populated we re-request with
        ``think=false`` to suppress the internal reasoning chain and get a
        direct answer.  The real Memento-Skills executor uses litellm's
        streaming interface; here we keep things simple with a single POST.
        """
        def _build_payload(thinking: bool) -> dict:
            payload: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                },
            }
            if not thinking:
                # Suppress the extended thinking chain for thinking-capable models.
                # Ollama supports this via the top-level "think" flag.
                payload["think"] = False
            return payload

        _log_event("LLM", f"→ POST /api/chat  model={self.model} …")
        t0 = time.time()
        try:
            # First attempt — let model think if it wants to
            resp = requests.post(self._endpoint,
                                 json=_build_payload(thinking=True),
                                 timeout=self.timeout)
            resp.raise_for_status()
            data    = resp.json()
            message = data.get("message", {})
            text    = (message.get("content") or "").strip()

            # Thinking models (Qwen3, QwQ …) put their final answer in
            # "content" and the scratchpad in "thinking".  If content is
            # empty the model spent all tokens on the scratchpad — retry
            # with thinking disabled.
            if not text:
                _log_event("LLM",
                    "content empty (thinking-model token budget exhausted) "
                    "— retrying with think=false")
                resp = requests.post(self._endpoint,
                                     json=_build_payload(thinking=False),
                                     timeout=self.timeout)
                resp.raise_for_status()
                data    = resp.json()
                message = data.get("message", {})
                text    = (message.get("content") or "").strip()

            elapsed = time.time() - t0
            _log_event("LLM", f"← response in {elapsed:.1f}s  "
                               f"({len(text)} chars)")
            return text
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s. "
                "Try a smaller model or increase --timeout."
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure 'ollama serve' is running on your machine."
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc


# ════════════════════════════════════════════════════════════════════════════
# SKILL STORE  (mirrors core/skill/store/)
# ════════════════════════════════════════════════════════════════════════════

class SkillStore:
    """In-memory skill library.  Real system: SQLite + sqlite-vec."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        _log_event("STORE", f"'{skill.name}' saved  "
                            f"(v{skill.version}, utility={skill.utility_score:.2f})")

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def summary_table(self) -> None:
        if RICH:
            t = Table(title="📚 Skill Library", box=box.ROUNDED, show_lines=True)
            t.add_column("Name",        style="cyan",    no_wrap=True)
            t.add_column("Mode",        style="magenta")
            t.add_column("Utility",     style="yellow",  justify="right")
            t.add_column("Version",     style="green",   justify="right")
            t.add_column("Description", style="white")
            for s in self._skills.values():
                t.add_row(
                    s.name,
                    s.execution_mode.name,
                    f"{s.utility_score:.2f}",
                    str(s.version),
                    textwrap.shorten(s.description, 55),
                )
            console.print(t)
        else:
            print("\n=== Skill Library ===")
            for s in self._skills.values():
                print(f"  {s.name:25s}  util={s.utility_score:.2f}  v{s.version}")


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL  (mirrors core/skill/retrieval/)
# ════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> list[str]:
    return text.lower().split()


def bm25_score(query: str, skill: Skill,
               k1: float = 1.5, b: float = 0.75,
               corpus_size: int = 10, avg_dl: float = 8.0) -> float:
    """
    Simplified BM25.  Real system: rank-bm25 + EmbeddingClient + sqlite-vec.
    """
    q_tokens = _tokenise(query)
    d_tokens = _tokenise(skill.description + " " + skill.name)
    dl = len(d_tokens)
    freq: dict[str, int] = {}
    for tok in d_tokens:
        freq[tok] = freq.get(tok, 0) + 1
    score = 0.0
    for qt in q_tokens:
        f = freq.get(qt, 0)
        idf = math.log((corpus_size - f + 0.5) / (f + 0.5) + 1)
        tf_norm = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_dl))
        score += idf * tf_norm
    return score


class SkillRouter:
    """BM25 retrieval over the local skill library."""

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    def search(self, query: str, top_k: int = 3) -> list[tuple[Skill, float]]:
        scored = [
            (skill, bm25_score(query, skill,
                               corpus_size=len(self._store.all_skills())))
            for skill in self._store.all_skills()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ════════════════════════════════════════════════════════════════════════════
# EXECUTOR  (mirrors core/skill/execution/executor.py)
# ════════════════════════════════════════════════════════════════════════════

# System prompt that mirrors the SKILL_EXECUTE_PROMPT template in executor.py.
# The real prompt is longer and injects tool schemas; we keep the logic.
_EXECUTE_SYSTEM = """\
You are a precise task-execution assistant.
You will be given a skill specification (SKILL.md) and a user query.
Follow the skill instructions exactly to solve the query.
Reply ONLY with the answer — no meta-commentary, no preamble.
If the skill instructions are visibly broken or incomplete, start your reply
with the exact token [SKILL_FAILED] followed by a one-sentence diagnosis.
"""

_EXECUTE_USER = """\
=== SKILL.md ===
{skill_content}
=== END SKILL.md ===

User query: {query}
"""

class SkillExecutor:
    """
    Builds the prompt from SKILL.md + query, calls Ollama, parses the result.

    Real flow (executor.py):
      1. Build prompt from SKILL.md + available tool schemas + query
      2. litellm call → LLM returns tool_calls or plain text
      3a. tool_calls  → execute builtin tools (bash, web-search …) in sandbox
      3b. code block  → run via `uv` sandbox
      3c. plain text  → return as knowledge answer
    Here we use plain chat (no tool calls) to keep the demo self-contained.
    """

    def __init__(self, llm: OllamaLLM) -> None:
        self._llm = llm

    def execute(self, skill: Skill, query: str) -> SkillExecutionOutcome:
        _log_event("EXECUTOR",
                   f"skill='{skill.name}'  query='{textwrap.shorten(query, 60)}'")

        system = _EXECUTE_SYSTEM
        user   = _EXECUTE_USER.format(
            skill_content=skill.content,
            query=query,
        )

        _wait_spinner("Calling Ollama …")
        raw = self._llm.complete(system, user)

        if raw.startswith("[SKILL_FAILED]"):
            diagnosis = raw[len("[SKILL_FAILED]"):].strip()
            _log_event("EXECUTOR", f"skill self-reported failure: {diagnosis}")
            return SkillExecutionOutcome(
                success=False,
                result=raw,
                error=diagnosis,
                skill_name=skill.name,
            )

        return SkillExecutionOutcome(
            success=True,
            result=raw,
            skill_name=skill.name,
        )


# ════════════════════════════════════════════════════════════════════════════
# REFLECTOR  (mirrors Reflect → Write phases)
# ════════════════════════════════════════════════════════════════════════════

_REFLECT_SYSTEM = """\
You are a skill-improvement agent for a self-evolving AI framework.
You will receive a SKILL.md that caused an execution failure and the error
message produced by the LLM executor.  Your job is to rewrite SKILL.md so
that the failure cannot recur.  Return ONLY the corrected SKILL.md content —
no commentary, no markdown fences.
"""

_REFLECT_USER = """\
=== FAILING SKILL.md ===
{skill_content}
=== END ===

Error reported by executor:
{error}

Rewrite the SKILL.md now, fixing the root cause.
"""

@dataclass
class ReflectionResult:
    should_update: bool
    improved_content: str
    reason: str


class Reflector:
    """
    Analyses a failed outcome and asks the LLM to rewrite the skill.

    In the real system this is another litellm call that locates the broken
    skill section, patches it, and optionally triggers `skill-creator` to
    mint a brand-new skill when the old one is beyond repair.
    """

    def __init__(self, llm: OllamaLLM) -> None:
        self._llm = llm

    def reflect(self, outcome: SkillExecutionOutcome,
                original_skill: Skill) -> ReflectionResult:
        _log_event("REFLECTOR",
                   f"analysing failure for '{outcome.skill_name}' …")

        if outcome.success:
            return ReflectionResult(
                should_update=True,
                improved_content=original_skill.content,
                reason="Execution succeeded — utility score bumped.",
            )

        # Ask the real LLM to patch the skill
        _wait_spinner("Reflector calling Ollama to rewrite skill …")
        improved = self._llm.complete(
            system_prompt=_REFLECT_SYSTEM,
            user_prompt=_REFLECT_USER.format(
                skill_content=original_skill.content,
                error=outcome.error or outcome.result,
            ),
        )

        _log_event("REFLECTOR", "skill rewrite received from LLM.")
        return ReflectionResult(
            should_update=True,
            improved_content=improved,
            reason=f"Skill rewritten after failure: {outcome.error}",
        )


# ════════════════════════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR  (mirrors core/memento_s/agent.py)
# ════════════════════════════════════════════════════════════════════════════

class MementoAgent:
    """
    Top-level orchestrator.  Implements Read → Execute → Reflect → Write.

    In agent.py the intent router first decides DIRECT / AGENTIC.
    AGENTIC triggers plan generation → skill search (Read) → execute →
    reflect → write-back.  We skip planning here and go straight to
    skill routing, which is the interesting part.
    """

    def __init__(self, store: SkillStore, router: SkillRouter,
                 executor: SkillExecutor, reflector: Reflector) -> None:
        self.store     = store
        self.router    = router
        self.executor  = executor
        self.reflector = reflector

    def read(self, query: str) -> Skill | None:
        _log_event("AGENT[READ]", f"query='{textwrap.shorten(query, 60)}'")
        candidates = self.router.search(query, top_k=3)
        if not candidates:
            return None
        best, score = candidates[0]
        _log_event("AGENT[READ]",
                   f"best match → '{best.name}'  BM25={score:.3f}")
        return best

    def execute(self, skill: Skill, query: str) -> SkillExecutionOutcome:
        return self.executor.execute(skill, query)

    def reflect_and_write(self, outcome: SkillExecutionOutcome,
                          skill: Skill) -> None:
        reflection = self.reflector.reflect(outcome, skill)
        if not reflection.should_update:
            return
        if outcome.success:
            skill.utility_score = min(skill.utility_score + 0.2, 5.0)
            _log_event("AGENT[WRITE]",
                       f"utility ↑ → {skill.utility_score:.2f}  (content unchanged)")
        else:
            skill.content      = reflection.improved_content
            skill.version     += 1
            skill.utility_score = max(skill.utility_score - 0.3, 0.0)
            _log_event("AGENT[WRITE]",
                       f"skill rewritten → v{skill.version}  "
                       f"utility={skill.utility_score:.2f}")
        self.store.add(skill)

    def run(self, query: str, max_retries: int = 1) -> str:
        for attempt in range(1, max_retries + 2):
            skill = self.read(query)
            if skill is None:
                return "No suitable skill found."
            outcome = self.execute(skill, query)
            self.reflect_and_write(outcome, skill)
            if outcome.success:
                return outcome.result
            _log_event("AGENT", f"attempt {attempt} failed — retrying …")
        return "Task failed after all retries."


# ════════════════════════════════════════════════════════════════════════════
# SKILL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def build_skill_store() -> SkillStore:
    store = SkillStore()

    store.add(Skill(
        name="text-summariser",
        description="Summarise long text documents articles research papers",
        content=textwrap.dedent("""\
            # text-summariser

            ## Purpose
            Condense any supplied document into a clear, concise summary.

            ## Instructions
            1. Read the full document provided in the query.
            2. Identify the main thesis, key arguments, and conclusions.
            3. Write a 3–5 sentence summary in neutral language.
            4. Do not include any information not present in the source text.
            5. End with: "Summary complete."
        """),
        execution_mode=ExecutionMode.KNOWLEDGE,
        utility_score=1.0,
    ))

    store.add(Skill(
        name="web-search",
        description="Search web internet retrieve current news information articles",
        content=textwrap.dedent("""\
            # web-search

            ## Purpose
            Help the user find up-to-date information from the internet.

            ## Instructions
            (In the real system this skill calls the web_search tool.
             In this demo, reason from your training data and note that
             actual live data is unavailable.)

            1. Understand what the user is looking for.
            2. Recall relevant recent facts from your training data.
            3. Present them clearly with source hints where possible.
            4. Acknowledge if information may be outdated.
        """),
        execution_mode=ExecutionMode.PLAYBOOK,
        utility_score=1.0,
    ))

    # ── deliberately broken skill for Example 2 ──────────────────────────
    store.add(Skill(
        name="finance-calculator",
        description="Calculate financial metrics compound interest ROI loan payments",
        content=textwrap.dedent("""\
            # finance-calculator  [VERSION 1 — BROKEN]

            ## Purpose
            Perform financial calculations.

            ## Instructions
            1. Identify the formula requested by the user.
            2. Use the simplified formula:  A = P * r * t
               (This is WRONG for compound interest — it is the simple
               interest formula.  The compounding frequency n is missing.)
            3. Return the result.

            NOTE: This skill is intentionally broken to demonstrate the
            self-evolution loop.  The executor will report [SKILL_FAILED].
        """),
        execution_mode=ExecutionMode.KNOWLEDGE,
        utility_score=0.6,
    ))

    return store


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _log_event(phase: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    if RICH:
        colour = {
            "STORE":       "dim cyan",
            "LLM":         "blue",
            "EXECUTOR":    "yellow",
            "REFLECTOR":   "magenta",
            "AGENT[READ]": "green",
            "AGENT[WRITE]":"red",
            "AGENT":       "white",
        }.get(phase, "white")
        console.print(f"[{colour}][{ts}] [{phase}][/{colour}] {msg}")
    else:
        print(f"[{ts}] [{phase}] {msg}")


def _wait_spinner(label: str) -> None:
    """Show a one-line spinner while the LLM is thinking (Rich only)."""
    if RICH:
        with Live(Spinner("dots", text=f" [dim]{label}[/dim]"),
                  refresh_per_second=10, transient=True):
            time.sleep(0.05)   # spinner shown; actual wait is inside LLM call
    # (no-op for plain console — the LLM call itself blocks)


def section(title: str) -> None:
    if RICH:
        console.print()
        console.rule(f"[bold]{title}[/bold]")
        console.print()
    else:
        print(f"\n{'═'*70}\n  {title}\n{'═'*70}")


def show_skill(skill: Skill) -> None:
    if RICH:
        console.print(Panel(
            Syntax(skill.content, "markdown", theme="monokai", word_wrap=True),
            title=f"[bold cyan]SKILL.md — {skill.name}  v{skill.version}[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print(f"\n--- SKILL.md ({skill.name} v{skill.version}) ---\n"
              f"{skill.content}\n---")


def show_result(result: str, title: str = "Agent Output") -> None:
    if RICH:
        console.print(Panel(result, title=f"[bold green]{title}[/bold green]",
                            border_style="green"))
    else:
        print(f"\n[{title}]\n{result}\n")


def show_scores(candidates: list[tuple[Skill, float]]) -> None:
    if RICH:
        t = Table(title="BM25 Retrieval Scores", box=box.SIMPLE)
        t.add_column("Rank", justify="center", style="dim")
        t.add_column("Skill",      style="cyan")
        t.add_column("BM25",       justify="right", style="yellow")
        t.add_column("Mode",       style="magenta")
        for i, (sk, sc) in enumerate(candidates, 1):
            t.add_row(str(i), sk.name, f"{sc:.4f}", sk.execution_mode.name)
        console.print(t)
    else:
        for i, (sk, sc) in enumerate(candidates, 1):
            print(f"  #{i}  {sk.name:25s}  BM25={sc:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1 — SKILL ROUTING
# ════════════════════════════════════════════════════════════════════════════

def example_skill_routing(agent: MementoAgent) -> None:
    section("EXAMPLE 1 — Skill Routing  (Read → Execute → Reflect → Write)")

    query = (
        "Summarise the following text:\n\n"
        "The transformer architecture, introduced by Vaswani et al. in 2017, "
        "replaced recurrent networks with self-attention mechanisms. "
        "This allowed parallelisation during training and led to dramatic "
        "improvements in NLP tasks. GPT, BERT, and their descendants all "
        "build on this foundation. By 2024, transformers had expanded beyond "
        "text into vision, audio, and multi-modal domains, becoming the "
        "dominant paradigm in deep learning."
    )

    if RICH:
        console.print(Panel(f"[bold]{textwrap.shorten(query, 120)}[/bold]",
                            title="User Query", border_style="white"))
    else:
        print(f"\nUser query: {textwrap.shorten(query, 120)}")

    # READ
    console.print("\n[bold]Step 1 — READ: BM25 retrieval[/bold]" if RICH
                  else "\nStep 1 — READ")
    candidates = agent.router.search(query, top_k=3)
    show_scores(candidates)
    best_skill = candidates[0][0]

    # EXECUTE
    console.print("\n[bold]Step 2 — EXECUTE: run best skill via Ollama[/bold]"
                  if RICH else "\nStep 2 — EXECUTE")
    show_skill(best_skill)
    outcome = agent.execute(best_skill, query)

    # REFLECT + WRITE
    console.print("\n[bold]Step 3 — REFLECT + WRITE[/bold]" if RICH
                  else "\nStep 3 — REFLECT + WRITE")
    agent.reflect_and_write(outcome, best_skill)
    show_result(outcome.result)


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2 — SELF-EVOLUTION LOOP
# ════════════════════════════════════════════════════════════════════════════

def example_self_evolution(agent: MementoAgent) -> None:
    section("EXAMPLE 2 — Self-Evolution Loop  (broken skill → LLM fixes it)")

    query = (
        "Calculate compound interest on €1000 at an annual rate of 5%, "
        "compounded quarterly, for 3 years.  Show the formula and result."
    )

    if RICH:
        console.print(Panel(f"[bold]{query}[/bold]",
                            title="User Query", border_style="white"))
    else:
        print(f"\nUser query: {query}")

    skill = agent.store.get("finance-calculator")
    assert skill is not None

    console.print("\n[bold]── BEFORE evolution ──[/bold]" if RICH
                  else "\n── BEFORE evolution ──")
    show_skill(skill)

    # ── Attempt 1 (expected to fail — skill is broken) ────────────────────
    console.print("\n[bold]Attempt 1  (broken skill)[/bold]" if RICH
                  else "\nAttempt 1 (broken skill)")
    outcome1 = agent.execute(skill, query)

    if outcome1.success:
        # Very capable models may push through despite broken instructions.
        # We still run the reflection step to show the write-back mechanism.
        console.print(
            "\n[dim yellow]⚠  The model answered despite broken instructions.\n"
            "Running reflection anyway to demonstrate the write-back.[/dim yellow]"
            if RICH else "\n[Warning] Model answered despite broken instructions."
        )

    # ── Reflect: LLM rewrites the skill ───────────────────────────────────
    console.print("\n[bold]Reflecting on outcome → Ollama rewrites SKILL.md …[/bold]"
                  if RICH else "\nReflecting …")
    agent.reflect_and_write(outcome1, skill)

    updated = agent.store.get("finance-calculator")
    console.print("\n[bold]── AFTER evolution ──[/bold]" if RICH
                  else "\n── AFTER evolution ──")
    show_skill(updated)

    # ── Attempt 2 (with improved skill) ──────────────────────────────────
    console.print("\n[bold]Attempt 2  (improved skill)[/bold]" if RICH
                  else "\nAttempt 2 (improved skill)")
    outcome2 = agent.execute(updated, query)
    agent.reflect_and_write(outcome2, updated)
    show_result(outcome2.result)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memento-Skills demo with real Ollama backend"
    )
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL})")
    p.add_argument("--url",     default=DEFAULT_URL,
                   help=f"Ollama base URL (default: {DEFAULT_URL})")
    p.add_argument("--timeout", default=300, type=int,
                   help="Request timeout in seconds (default: 300)")
    p.add_argument("--example", choices=["1", "2", "both"], default="both",
                   help="Which example to run (default: both)")
    p.add_argument("--save-html", metavar="FILE",
                   help="Save full Rich-coloured output as self-contained HTML file")
    p.add_argument("--save-md",   metavar="FILE",
                   help="Save plain-text output as Markdown file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── set up optional file recorders ────────────────────────────────────
    # We create a Rich Console that records everything, then at the end
    # export to HTML and/or Markdown alongside the live screen output.
    global console
    if RICH:
        from rich.console import Console
        from io import StringIO

        record = args.save_html or args.save_md
        if record:
            # file_console records markup; screen_console renders to terminal
            file_console   = Console(record=True, highlight=True)
            screen_console = console          # original terminal console

            # Monkey-patch module-level console so all helpers write to both
            class _TeeConsole:
                """Forwards every .print() / .rule() call to both consoles."""
                def print(self, *a, **kw):
                    screen_console.print(*a, **kw)
                    file_console.print(*a, **kw)
                def rule(self, title="", **kw):
                    screen_console.rule(title, **kw)
                    file_console.rule(title, **kw)

            console = _TeeConsole()
        else:
            file_console = None
    else:
        file_console = None

    # ── banner ────────────────────────────────────────────────────────────
    if RICH:
        console.print(Panel.fit(
            "[bold white]Memento-Skills Demo[/bold white]  "
            "[dim](Ollama edition)[/dim]\n"
            "[dim]https://github.com/Memento-Teams/Memento-Skills[/dim]\n\n"
            f"Model : [cyan]{args.model}[/cyan]\n"
            f"Server: [cyan]{args.url}[/cyan]",
            border_style="bright_blue",
            title="[bright_blue]★ Self-Evolving Agent Framework ★[/bright_blue]",
        ))
    else:
        print("=" * 70)
        print("  Memento-Skills Demo (Ollama edition)")
        print(f"  Model : {args.model}")
        print(f"  Server: {args.url}")
        print("=" * 70)

    # ── verify Ollama is reachable ─────────────────────────────────────────
    llm = OllamaLLM(model=args.model, base_url=args.url, timeout=args.timeout)
    if not llm.ping():
        console.print(
            f"\n[bold red]ERROR:[/bold red] Cannot reach Ollama at [cyan]{args.url}[/cyan].\n"
            "Make sure Ollama is running:  [dim]ollama serve[/dim]"
            if RICH else
            f"\nERROR: Cannot reach Ollama at {args.url}.\n"
            "Make sure Ollama is running: ollama serve"
        )
        sys.exit(1)

    available = llm.available_models()
    if args.model not in available:
        console.print(
            f"\n[bold yellow]WARNING:[/bold yellow] model '[cyan]{args.model}[/cyan]' "
            f"not found.  Available:\n  " + "\n  ".join(available)
            if RICH else
            f"\nWARNING: model '{args.model}' not found.\nAvailable: {available}"
        )
        sys.exit(1)

    _log_event("LLM", f"Ollama reachable  model={args.model}")

    # ── wire up components ─────────────────────────────────────────────────
    store     = build_skill_store()
    router    = SkillRouter(store)
    executor  = SkillExecutor(llm)
    reflector = Reflector(llm)
    agent     = MementoAgent(store, router, executor, reflector)

    section("Initial Skill Library")
    store.summary_table()

    if args.example in ("1", "both"):
        example_skill_routing(agent)

    if args.example in ("2", "both"):
        example_self_evolution(agent)

    section("Final Skill Library State")
    store.summary_table()

    if RICH:
        console.print()
        console.rule("[bold bright_blue]Demo complete[/bold bright_blue]")
        console.print(
            "\n[dim]The real system adds: litellm multi-provider routing, "
            "SQLite + sqlite-vec skill store, BM25 + embedding hybrid recall, "
            "a remote skill catalogue, local uv sandbox execution, "
            "and a full GUI (Flet).[/dim]"
        )

    # ── export files ──────────────────────────────────────────────────────
    if RICH and file_console is not None:
        model_slug = args.model.replace("/", "-").replace(":", "-")

        if args.save_html:
            html_path = args.save_html
            # Rich exports a fully self-contained HTML with inline CSS/colours
            file_console.save_html(html_path, clear=False)
            screen_console.print(
                f"\n[green]HTML saved →[/green] [cyan]{html_path}[/cyan]"
            )

        if args.save_md:
            md_path = args.save_md
            # Rich exports plain text (no ANSI codes); we wrap it in a
            # Markdown code block with a header showing model + timestamp.
            plain_text = file_console.export_text(clear=False)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            md_content = (
                f"# Memento-Skills Demo — {args.model}\n"
                f"_Run at {ts}_\n"
                f"_Server: {args.url}_\n\n"
                "```\n"
                + plain_text +
                "```\n"
            )
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(md_content)
            screen_console.print(
                f"[green]Markdown saved →[/green] [cyan]{md_path}[/cyan]"
            )


if __name__ == "__main__":
    main()
