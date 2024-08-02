import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_executor import ToolExecutor
from tools.search import search_tool
from tools.scrape import scrape_tool
from tools.summary import summary_tool
from langchain_core.messages import FunctionMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import streamlit as st
from typing import Annotated, Literal, TypedDict
import json

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

system_message = SystemMessage(
    content="""
    You are a world-class researcher who can do detailed research on any topic and produce fact-based results. You do not make things up. You will try as hard as possible to gather facts and data to back up your research.

    Follow these steps for your research:
    1. First read the user's query carefully and search for relevant information on the internet.
    2. You must use scrape_website tool to extract the main content from a website.
    3. You must Use the summarize tool to summarize long pieces of text based on a given objective.
    4. Always provide a detailed summary of your all your research findings from all the scraped data.
    5. Provide references to the sources you used in your research
    6. You must use the tools provided to help you with your research.
    7. You must always provide a fact-based response to the user's query.
    8. Your final response MUST be formatted exactly as follows:

    Answer: [Provide an answer to the user's query based on all the information gathered, always provide facts and numbers, don't make things up and don't provide opinions.]

    Detailed Summary:
    [Provide a detailed summary of your research findings, at least 200 words long, focusing on facts and numbers. Do not make things up.]

    References:
    1. [Website Title 1]
    2. [Website Title 2]
    ...

    This format is crucial. Always include all three sections: Answer, Detailed Summary, and references.

    Use the provided tools (scrape_website, search and summarize) to accomplish these tasks.
    """
)

tools = [search_tool, scrape_tool, summary_tool]

tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key).bind_tools(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # Log the model's thought process
    st.write("🤔 Agent's Thought Process:")
    st.write(response.content)
    if response.tool_calls:
        st.write("🛠️ Agent is using a tool:")
        for tool_call in response.tool_calls:
            # Check if tool_call is a dictionary
            if isinstance(tool_call, dict):
                tool_name = tool_call.get('name', 'Unknown tool')
                tool_args = json.dumps(tool_call.get('arguments', {}))
                st.write(f"- {tool_name}: {tool_args}")
            else:
                # Fallback for other structures
                st.write(f"- Tool call: {tool_call}")
    return {"messages": [response]}

def flow(query: str):
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    st.write("🔍 Starting research process...")
    final_state = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": 42}},
    )
    st.write("✅ Research process complete!")
    return final_state["messages"][-1].content

def main():
    st.set_page_config(page_title="TUSHAR'S AI BUDDY", page_icon="🔍")

    st.header("TUSHAR'S AI BUDDY")
    st.subheader("A tool to help you with your research")

    user_input = st.text_input("Enter your research objective")

    if user_input:
        st.write("Your research objective: ", user_input)

        with st.spinner("Researching..."):
            result = flow(user_input)

        st.success("Research complete!")
        st.info(result)

if __name__ == "__main__":
    main()