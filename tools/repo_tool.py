# tools/repo_tool.py

import os
import json
import subprocess
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class RepoToolInput(BaseModel):
    repo_url: str = Field(..., description="Git repository URL to clone or pull")
    local_path: str = Field("local_repo", description="Local folder to store the repo")

class RepoToolOutput(BaseModel):
    local_repo_path: str

class RepoTool(BaseTool):
    name: str = "repo_tool"
    description: str = (
        "Clones or updates a Git repository. "
        "Input must be a JSON string with 'repo_url' and optional 'local_path'. "
        "Output is JSON with 'local_repo_path'."
    )

    # Tools typically have a single string input, so we parse/return JSON
    def _run(self, tool_input: str) -> str:
        """
        Expects JSON: {"repo_url": "...", "local_path": "..."}
        Returns JSON: {"local_repo_path": "..."}
        """
        try:
            data = json.loads(tool_input)
            parsed = RepoToolInput(**data)
        except Exception as e:
            return json.dumps({"error": f"Invalid input format: {str(e)}"})

        repo_url = parsed.repo_url
        local_path = parsed.local_path

        if not os.path.exists(local_path):
            # Clone
            subprocess.run(["git", "clone", repo_url, local_path], check=True)
        else:
            # If offline, skip or handle errors; for now we do a git pull
            subprocess.run(["git", "-C", local_path, "pull"], check=True)

        output = RepoToolOutput(local_repo_path=os.path.abspath(local_path))
        return output.json()

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("Async not implemented.")

