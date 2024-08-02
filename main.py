# main.py
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

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

system_message = SystemMessage(
    content="""
    You are a world-class researcher who can do detailed research on any topic and produce fact-based results. You do not make things up. You will try as hard as possible to gather facts and data to back up your research.

    Follow these steps for your research:
    1. Search for relevant links related to the given objective.
    2. For each relevant link, extract the page content.
    3. If the extracted content has more than 10,000 words, summarize it.
    4. Combine all the information (original or summarized) to create a final summary aligned with the initial objective.
    5. Include references to the sources you've used in your final summary.
    6. You must use the tools provided to help you with your research.
    7. You must always provide a fact-based response to the user's query.
    8. You must always give three things, the answer to the user's query, the source of the information, and a summary of the information, ITS MUST TO QUALIFY AS A VALID RESPONSE.
    9. Summarize must be fact and number based and atleast 100 words, don't make things up.
    10. FINALLY FORMAT THE RESPONSE AS ANSWER, SOURCE, SUMMARY.

    Use the provided tools (Search, ScrapeWebsite, and Summarize) to accomplish these tasks.
    
    """
)

tools = [search_tool, scrape_tool, summary_tool]

tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key).bind_tools(tools=tools)  # type: ignore


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def flow(query: str):

    # Define a new graph
    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

    # Use the Runnable
    final_state = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": 42}},
    )
    return final_state["messages"][-1].content


def main():
    st.set_page_config(page_title="TUSHAR'S AI BUDDY", page_icon="🔍")

    st.header("TUSHAR'S AI BUDDY")
    st.subheader("A tool to help you with your research")

    user_input = st.text_input("Enter your research objective")

    if user_input:
        st.write("Your research objective: ", user_input)

        result = flow(user_input)

        st.info(result)


if __name__ == "__main__":
    main()
