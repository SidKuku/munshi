# tools/summarizer_tool.py

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.llms.base import LLM
from pydantic import Extra

class SummarizerToolInput(BaseModel):
    """Expects a list of code entities in JSON form."""
    entities: List[Dict[str, Any]] = Field(..., description="List of code entities to summarize")

class SummarizerToolOutput(BaseModel):
    """Returns the updated entities, each with a 'summary' field."""
    summarized_entities: List[Dict[str, Any]]

class SummarizerTool(BaseTool):
    name: str = "summarizer_tool"
    description: str = (
        "Tool for summarizing code snippets using an LLM. "
        "Input must be JSON with 'entities': [...]. Output is JSON with 'summarized_entities'."
    )

    # Must declare fields as pydantic if we store them on the class
    llm: LLM = Field(default=None, exclude=True)

    class Config:
        extra = Extra.allow  # or Extra.forbid, depending on preference

    def _run(self, tool_input: str) -> str:
        """
        Expects JSON: {"entities": [...code entities...] }
        Returns JSON: {"summarized_entities": [... updated with 'summary' ...]}
        """
        try:
            data = json.loads(tool_input)
            parsed_input = SummarizerToolInput(**data)
        except Exception as e:
            return json.dumps({"error": f"Invalid input: {str(e)}"})

        entities = parsed_input.entities
        updated = []

        for e in entities:
            code = e.get("code", "")
            docstring = e.get("docstring", "")
            entity_type = e.get("type", "unknown")
            entity_name = e.get("name", "unknown")

            prompt = f"""
You are an expert software engineer. Summarize the following {entity_type} named '{entity_name}' 
and explain its purpose and parameters.

Docstring (if any): {docstring}

Code:
{code}

Be concise, but include key details.
"""

            # We call self.llm, which is a HuggingFacePipeline-based LLM
            summary = self.llm(prompt)
            e["summary"] = summary
            updated.append(e)

        output = SummarizerToolOutput(summarized_entities=updated)
        return output.json()

    async def _arun(self, tool_input: str) -> str:
        raise NotImplementedError("Async not implemented.")

