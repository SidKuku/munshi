# tools/doc_builder_tool.py
import os
from langchain.tools import BaseTool

class DocBuilderTool(BaseTool):
    name = "doc_builder_tool"
    description = "Tool for building a Markdown documentation file from summarized entities."

    def _run(self, entities: list, output_path: str = "docs/Documentation.md") -> str:
        """
        Builds a Markdown file from the entities, writes it to 'output_path',
        and returns the path to the generated documentation.
        """
        # Ensure docs folder exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        md = "# Project Documentation\n\n"
        for e in entities:
            md += f"## {e['type'].title()}: {e['name']}\n\n"
            md += f"**Location**: `{e.get('file_path', '')}`\n\n"
            md += f"**Summary**:\n\n{e.get('summary', 'No summary available')}\n\n"
            md += "---\n\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        return os.path.abspath(output_path)

    async def _arun(self, entities: list, output_path: str = "docs/Documentation.md") -> str:
        """Async version not implemented."""
        raise NotImplementedError("Async run not implemented.")

