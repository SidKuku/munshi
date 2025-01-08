# tools/summarizer_tool.py
from langchain.tools import BaseTool
from langchain.llms.base import LLM

class SummarizerTool(BaseTool):
    name = "summarizer_tool"
    description = "Tool for summarizing code snippets using an LLM."

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    def _run(self, entities: list) -> list:
        """
        Takes a list of code entities, uses the LLM to summarize each, 
        and returns an updated list with 'summary' fields.
        """
        updated = []
        for e in entities:
            prompt = self._build_prompt(e)
            # If using a pipeline LLM, call it with prompt=prompt (and possibly max_new_tokens=512, etc.)
            summary = self.llm(prompt)
            e["summary"] = summary
            updated.append(e)
        return updated

    async def _arun(self, entities: list) -> list:
        """Async version (not implemented)."""
        raise NotImplementedError("Async run not implemented.")

    def _build_prompt(self, entity) -> str:
        """
        Construct a prompt for summarizing the entity.
        """
        code = entity.get("code", "")
        docstring = entity.get("docstring", "")
        entity_type = entity["type"]
        entity_name = entity["name"]

        prompt_template = f"""
You are an expert software engineer. Summarize the following {entity_type} named '{entity_name}' 
and explain its purpose and parameters.

Docstring (if any): {docstring}

Code:
{code}

Be concise, but include key details.
"""
        return prompt_template

