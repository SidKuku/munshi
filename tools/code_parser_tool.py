# tools/code_parser_tool.py
import os
import ast
from langchain.tools import BaseTool

class CodeParserTool(BaseTool):
    name = "code_parser_tool"
    description = "Tool for parsing Python code files and extracting classes/methods/docstrings."

    def _run(self, repo_path: str) -> list:
        """
        Walks the given repo_path, parses .py files, 
        returns a list of extracted entities with code snippets.
        """
        entities = []
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
                                doc = ast.get_docstring(node)
                                code_snippet = self._extract_source_segment(code, node)
                                entities.append({
                                    "type": "class",
                                    "name": name,
                                    "docstring": doc,
                                    "code": code_snippet,
                                    "file_path": file_path
                                })
                            elif isinstance(node, ast.FunctionDef):
                                name = node.name
                                doc = ast.get_docstring(node)
                                code_snippet = self._extract_source_segment(code, node)
                                entities.append({
                                    "type": "function",
                                    "name": name,
                                    "docstring": doc,
                                    "code": code_snippet,
                                    "file_path": file_path
                                })
                    except Exception:
                        # Ignore parse errors
                        pass
        return entities

    async def _arun(self, repo_path: str) -> list:
        """Async version (not implemented)."""
        raise NotImplementedError("Async run not implemented.")

    def _extract_source_segment(self, full_code: str, node) -> str:
        """
        Extract the exact source code for the given AST node (Python 3.8+).
        Fallback if ast.get_source_segment is unavailable or fails.
        """
        try:
            return ast.get_source_segment(full_code, node)
        except:
            # Fallback approach:
            lines = full_code.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])

