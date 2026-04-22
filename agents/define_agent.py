from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from .backend import backend
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from .model import _EXECUTE_SYSTEM, _REFLECT_SYSTEM, SkillExecutionOutput

load_dotenv('.env')

model = ChatOpenAI(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

middleware = [
    SkillsMiddleware(backend=backend, sources=["/skills/"]),
    FilesystemMiddleware(backend=backend)
]

read_agent = create_agent(
    model = model,
    system_prompt=_EXECUTE_SYSTEM,
    middleware=middleware
)

write_agent = create_agent(
    model=model,
    middleware=middleware
)