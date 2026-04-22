from pydantic import BaseModel, Field
from typing import List, Optional

class ErrorInfo(BaseModel):
    type: str = Field(..., description="Exception type (e.g., ConnectionError, FileNotFoundError)")
    message: str = Field(..., description="Short human-readable error message")
    raw: str = Field(..., description="Truncated raw stderr or traceback (max ~300 chars)")

class TraceStep(BaseModel):
    step: int = Field(..., description="The sequential step number in the execution trace.")
    action: str = Field(..., description="The type of action taken (e.g., 'route_skill', 'invoke_tool').")
    detail: str = Field(..., description="A concise description of what was done in this step.")
    evidence: str = Field(..., description="Supporting evidence for this step (e.g., tool call ID, filename, or short proof).")
    
    error: Optional[ErrorInfo] = Field(
        default=None,
        description="Structured error info if this step failed"
    )

class SkillExecutionOutput(BaseModel):
    success: bool
    result: str
    error: Optional[str]
    skill_name: str
    trace: List[TraceStep]
    
_EXECUTE_SYSTEM = """\
You are a precise task-execution assistant.
You will be given a skill specification (SKILL.md) and a user query.
Follow the skill instructions exactly to solve the query.
Reply ONLY with the answer — no meta-commentary, no preamble.
If the skill instructions are visibly broken or incomplete, start your reply
with the exact token [SKILL_FAILED] followed by a one-sentence diagnosis.
"""

_REFLECT_SYSTEM = """\
You are a skill-improvement agent for a self-evolving AI framework.
You will receive a SKILL.md that caused an execution failure and the error
message produced by the LLM executor.  Your job is to rewrite SKILL.md so
that the failure cannot recur.  Return ONLY the corrected SKILL.md content —
no commentary, no markdown fences.
"""