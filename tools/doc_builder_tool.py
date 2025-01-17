# tools/doc_builder_tool.py

import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class DocBuilderToolInput(BaseModel):
    entities: List[Dict[str, Any]] = Field(..., description="List of summarized entities")
    output_path: str = Field("docs/Documentation.md", description="Where to write the Markdown doc")

class DocBuilderToolOutput(BaseModel):
    doc_path: str

class DocBuilderTool(BaseTool):
    name: str = "doc_builder_tool"
    description: str = (
        "Creates a Markdown documentation file from summarized entities. "
        "Input must be JSON with 'entities' and 'output_path'. "
        "Output is JSON with 'doc_path'."
    )

    def _run(self, tool_input: str) -> str:
        """
        Expects JSON: {"entities": [...], "output_path": "..."}
        Returns JSON: {"doc_path": "..."}
        """
        try:
            data = json.loads(tool_input)
            parsed_input = DocBuilderToolInput(**data)
        except Exception as e:
            return json.dumps({"error": f"Invalid input: {str(e)}"})

        output_dir = os.path.dirname(parsed_input.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        md_content = "# Project Documentation\n\n"

        for e in parsed_input.entities:
            e_type = e.get("type", "").title()
            e_name = e.get("name", "")
            e_file = e.get("file_path", "")
            summary = e.get("summary", "No summary available")

            md_content += f"## {e_type}: {e_name}\n\n"
            md_content += f"**Location**: `{e_file}`\n\n"
            md_content += f"**Summary**:\n\n{summary}\n\n"
            md_content += "---\n\n"

        with open(parsed_input.output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        output = DocBuilderToolOutput(doc_path=os.path.abspath(parsed_input.output_path))
        return output.json()

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("Async not implemented.")

