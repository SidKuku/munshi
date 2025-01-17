# main.py

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent

# Tools
from tools.repo_tool import RepoTool
from tools.code_parser_tool import CodeParserTool
from tools.summarizer_tool import SummarizerTool
from tools.doc_builder_tool import DocBuilderTool

def load_local_llm(model_path: str):
    """
    Load a local Hugging Face model pipeline from the given path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    local_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
#        device=0,
        max_new_tokens=512,  # adjust as needed
        temperature=0.1
    )
    return local_pipeline

def create_langchain_llm(hf_pipeline):
    """
    Wrap the HF pipeline in a LangChain LLM interface.
    """
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm

def main():
    # 1. Load local model
    model_path = os.path.join("local_model")  # path to your local StarCoder or other model
    hf_pipeline = load_local_llm(model_path)
    langchain_llm = create_langchain_llm(hf_pipeline)

    # 2. Create the Tools
    repo_tool = RepoTool()
    parser_tool = CodeParserTool()
    summarizer_tool = SummarizerTool(llm=langchain_llm)
    doc_builder_tool = DocBuilderTool()

    # 3. Wrap each in a LangChain Tool
    #    Each tool has a single input (string), so we’ll pass/return JSON to handle structured data.
    repo_tool_lc = Tool(
        name=repo_tool.name,
        func=repo_tool.run,  # or repo_tool._run
        description=repo_tool.description
    )
    parser_tool_lc = Tool(
        name=parser_tool.name,
        func=parser_tool.run,
        description=parser_tool.description
    )
    summarizer_tool_lc = Tool(
        name=summarizer_tool.name,
        func=summarizer_tool.run,
        description=summarizer_tool.description
    )
    doc_builder_tool_lc = Tool(
        name=doc_builder_tool.name,
        func=doc_builder_tool.run,
        description=doc_builder_tool.description
    )

    tools_list = [repo_tool_lc, parser_tool_lc, summarizer_tool_lc, doc_builder_tool_lc]

    # 4. Create ZeroShotAgent Prompt
    from langchain.prompts import PromptTemplate

    prefix = """You are a software documentation agent. 
You have access to the following tools:"""
    suffix = """Begin!"""

    format_instructions = """Use the following format:
Thought: what you think to do
Action: one of [repo_tool, code_parser_tool, summarizer_tool, doc_builder_tool]
Action Input: a JSON string with the required fields for that tool
Observation: the result of the action (which may be JSON)
... (repeat Thought/Action/Observation as needed)
Thought: I now have the final answer
Final Answer: the final result of all operations
"""

    prompt_template = ZeroShotAgent.create_prompt(
        tools=tools_list,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions
    )

    # 5. Build an LLMChain for the agent
    llm_chain = LLMChain(llm=langchain_llm, prompt=prompt_template)

    # 6. Create the ZeroShotAgent with that chain
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=[t.name for t in tools_list],  # which tools can be called
        verbose=True
    )

    # 7. Build an AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools_list,
        verbose=True
    )

    # 8. Provide the user query
    user_query = """
Please clone the repo at https://github.com/SidKuku/cgm-remote-monitor.git into a local folder,
parse the code, summarize it, and build the documentation in 'docs/Documentation.md'.
"""

    result = agent_executor.run(user_query)
    print("Agent result:", result)

if __name__ == "__main__":
    main()

