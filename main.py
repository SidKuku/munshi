# main.py
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate

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
        torch_dtype="auto",  # or torch.float16 if supported
        device_map="auto"
    )
    local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return local_pipeline

def create_langchain_llm(hf_pipeline):
    """
    Wrap the HF pipeline in a LangChain LLM interface.
    """
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm

def main():
    # 1. Load local HF pipeline & wrap it in LangChain LLM
    model_path = os.path.join("local_model")  # path to your local model
    hf_pipeline = load_local_llm(model_path)
    llm = create_langchain_llm(hf_pipeline)

    # 2. Instantiate Tools
    repo_tool = RepoTool()
    parser_tool = CodeParserTool()
    summarizer_tool = SummarizerTool(llm=llm)
    doc_builder_tool = DocBuilderTool()

    # 3. Convert Tools to LangChain "Tool" objects
    langchain_tools = [
        Tool(
            name=repo_tool.name,
            func=repo_tool.run,
            description=repo_tool.description
        ),
        Tool(
            name=parser_tool.name,
            func=parser_tool.run,
            description=parser_tool.description
        ),
        Tool(
            name=summarizer_tool.name,
            func=summarizer_tool.run,
            description=summarizer_tool.description
        ),
        Tool(
            name=doc_builder_tool.name,
            func=doc_builder_tool.run,
            description=doc_builder_tool.description
        ),
    ]

    # 4. Create the prompt for a ZeroShotAgent
    prefix = """You are a software documentation agent. 
You have access to the following tools:"""
    suffix = """Begin!"""

    format_instructions = """Use the following format:
Thought: what you think to do
Action: one of [repo_tool, code_parser_tool, summarizer_tool, doc_builder_tool]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now have the final answer
Final Answer: the final result of all operations
"""

    prompt = ZeroShotAgent.create_prompt(
        tools=langchain_tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions
    )

    agent = ZeroShotAgent(llm=llm, tools=langchain_tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=langchain_tools)

    # 5. Sample user query:
    user_query = """
    Please clone the repo at https://github.com/some/repo.git into a local folder,
    parse the code, summarize it, and build the documentation in 'docs/Documentation.md'.
    """

    # 6. Run the agent
    result = agent_executor.run(user_query)
    print("Agent result:", result)

if __name__ == "__main__":
    main()

