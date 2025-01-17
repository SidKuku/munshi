# tools/code_parser_tool.py

import os
import json
import ast
from typing import List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class ParserToolInput(BaseModel):
    repo_path: str = Field(..., description="Path to the local repository to parse")

class Entity(BaseModel):
    type: str
    name: str
    docstring: str = ""
    code: str
    file_path: str

class ParserToolOutput(BaseModel):
    entities: List[Entity]

class CodeParserTool(BaseTool):
    name: str = "code_parser_tool"
    description: str = (
        "Parses Python files in a repo and extracts classes/functions. "
        "Input must be JSON: {'repo_path': '...'}, output is JSON with 'entities'."
    )

    def _run(self, tool_input: str) -> str:
        """
        Expects JSON input: {"repo_path": "..."}
        Returns JSON: {"entities": [ ... ]}
        """
        try:
            data = json.loads(tool_input)
            parsed = ParserToolInput(**data)
        except Exception as e:
            return json.dumps({"error": f"Invalid input: {str(e)}"})

        repo_path = parsed.repo_path
        entities: List[Dict[str, Any]] = []

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    try:
                        tree = ast.parse(code)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                name = node.name
                                doc = ast.get_docstring(node) or ""
                                code_snippet = self._extract_code_segment(code, node)
                                entities.append({
                                    "type": "class",
                                    "name": name,
                                    "docstring": doc,
                                    "code": code_snippet,
                                    "file_path": file_path
                                })
                            elif isinstance(node, ast.FunctionDef):
                                name = node.name
                                doc = ast.get_docstring(node) or ""
                                code_snippet = self._extract_code_segment(code, node)
                                entities.append({
                                    "type": "function",
                                    "name": name,
                                    "docstring": doc,
                                    "code": code_snippet,
                                    "file_path": file_path
                                })
                    except:
                        pass
        
        # Wrap in pydantic model
        output = ParserToolOutput(entities=[Entity(**e) for e in entities])
        return output.json()

    def _extract_code_segment(self, full_code: str, node) -> str:
        # For Python 3.8+: ast.get_source_segment
        try:
            return ast.get_source_segment(full_code, node)
        except:
            lines = full_code.splitlines()
            return "\n".join(lines[node.lineno - 1: node.end_lineno])

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("Async not implemented.")

