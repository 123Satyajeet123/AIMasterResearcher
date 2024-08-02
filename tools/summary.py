# tools/summary_tool.py
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class SummaryInput(BaseModel):
    objective: str = Field(..., description="The research objective")
    content: str = Field(..., description="The content to summarize")
    url: str = Field(..., description="The source URL of the content")


def summarize(objective: str, content: str, url: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)  # type: ignore
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Summarize the following text based on the objective: "{objective}"
    Text: "{text}"
    Summary:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["objective", "text"]
    )
    combine_prompt = """
    Combine the following summaries into a coherent summary that addresses the objective: "{objective}"
    Summaries: "{text}"
    Final Summary:
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["objective", "text"]
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=True,
    )
    output = summary_chain.run(input_documents=docs, objective=objective)

    # print(f"Summarized content: {output}")

    return {"summary": output, "source_url": url}


# content = scrape_website("https://ai.meta.com/blog/meta-llama-3-1/")["content"]

# summarize("Latest version of llama3 model?",content, "https://ai.meta.com/blog/meta-llama-3-1/")


summary_tool = Tool(
    name="Summarize",
    func=summarize,
    description="Summarizes long pieces of text based on a given objective.",
    args_schema=SummaryInput,
)
