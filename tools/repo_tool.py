# tools/repo_tool.py
import os
import subprocess
from langchain.tools import BaseTool

class RepoTool(BaseTool):
    name = "repo_tool"
    description = "Tool for cloning or updating a Git repository."

    def _run(self, repo_url: str, local_path: str = "local_repo") -> str:
        """
        Clones or updates the repository at the given path.
        Returns the path to the local repo.
        """
        if not os.path.exists(local_path):
            # Clone
            subprocess.run(["git", "clone", repo_url, local_path], check=True)
        else:
            # If not truly offline, you can do a pull. 
            # If offline, skip or handle differently.
            subprocess.run(["git", "-C", local_path, "pull"], check=True)

        return os.path.abspath(local_path)

    async def _arun(self, repo_url: str, local_path: str = "local_repo") -> str:
        """Asynchronous version (not implemented)."""
        raise NotImplementedError("Async run not implemented.")

