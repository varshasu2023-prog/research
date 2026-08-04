import os
import streamlit as st
from typing import Annotated, TypedDict, List, Dict
from typing_extensions import Literal

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END


# ==========================================================
# 1. API KEY INITIALIZATION
# ==========================================================

try:
    groq_bec_api_key = st.secrets["GROQ_BEC_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]

    os.environ["GROQ_API_KEY"] = groq_bec_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

except Exception as e:
    st.error(f"API Key Error: {e}")
    st.stop()


# ==========================================================
# 2. STATE DEFINITION
# ==========================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    domain: str
    basic_problem: str
    research_gaps: List[str]
    paper_draft: Dict[str, str]
    next_step: str



# ==========================================================
# 3. GROQ INITIALIZATION
# ==========================================================

llm = ChatGroq(
    api_key=groq_bec_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=2
)


search_tool = TavilySearchResults(
    tavily_api_key=tavily_api_key,
    max_results=3
)

arxiv_tool = ArxivAPIWrapper()



# ==========================================================
# 4. AGENT CREATION FUNCTION
# ==========================================================

def create_agent(llm, tools, system_prompt):

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])

    if tools:
        return prompt | llm.bind_tools(tools)

    return prompt | llm



# ==========================================================
# 5. NODE DEFINITIONS
# ==========================================================

def explorer_node(state: AgentState):

    prompt = f"""
    You are a Research Explorer.

    Domain:
    {state['domain']}

    Problem:
    {state['basic_problem']}

    Find 3 important recent research papers.
    Focus on methods and architectures.
    """

    agent = create_agent(
        llm,
        [search_tool],
        prompt
    )

    response = agent.invoke({
        "messages": state["messages"]
    })

    return {
        "messages":[response],
        "next_step":"reviewer"
    }



def reviewer_node(state: AgentState):

    prompt = f"""
    You are a Literature Reviewer.

    Identify 3 research gaps in:

    Domain:
    {state['domain']}

    Problem:
    {state['basic_problem']}
    """

    agent = create_agent(
        llm,
        None,
        prompt
    )

    response = agent.invoke({
        "messages":state["messages"]
    })


    gaps = [
        x for x in response.content.split("\n")
        if "gap" in x.lower()
    ][:3]


    return {
        "messages":[response],
        "research_gaps":gaps,
        "next_step":"writer"
    }




def writer_node(state: AgentState):

    prompt = f"""
    You are an Academic Paper Writer.

    Research gaps:
    {state['research_gaps']}

    Generate:

    1. Abstract
    2. Introduction
    3. Methodology
    4. Proposed Architecture
    5. Conclusion

    Use formal academic language.
    """

    agent=create_agent(
        llm,
        None,
        prompt
    )


    response=agent.invoke({
        "messages":state["messages"]
    })


    return {
        "messages":[response],
        "paper_draft":{
            "full_report":response.content
        },
        "next_step":"end"
    }



# ==========================================================
# 6. LANGGRAPH CONSTRUCTION
# ==========================================================

def router(state:AgentState):
    return state["next_step"]


workflow = StateGraph(AgentState)


workflow.add_node(
    "explorer",
    explorer_node
)

workflow.add_node(
    "reviewer",
    reviewer_node
)

workflow.add_node(
    "writer",
    writer_node
)


workflow.add_edge(
    START,
    "explorer"
)


workflow.add_conditional_edges(
    "explorer",
    router,
    {
        "reviewer":"reviewer",
        "end":END
    }
)


workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "writer":"writer",
        "end":END
    }
)


workflow.add_edge(
    "writer",
    END
)


app = workflow.compile()



# ==========================================================
# 7. STREAMLIT UI
# ==========================================================

st.title("⚡ GROQ Research Explorer")

domain = st.text_input(
    "Enter Research Domain",
    "Sustainable Energy AI"
)


problem = st.text_area(
    "Enter Research Problem",
    "Predicting micro-grid stability with renewable energy"
)



if st.button("Start Research"):

    initial_input = {

        "domain":domain,

        "basic_problem":problem,

        "messages":[
            HumanMessage(
                content="Analyze research papers."
            )
        ],

        "research_gaps":[],

        "paper_draft":{},

        "next_step":"explorer"
    }


    st.info("Research started...")


    for output in app.stream(initial_input):

        for node,value in output.items():

            st.subheader(
                f"Node: {node}"
            )


            if "messages" in value:

                st.write(
                    value["messages"][-1].content
                )


            if "research_gaps" in value:

                st.write(
                    "Research Gaps:"
                )

                st.write(
                    value["research_gaps"]
                )


            if "paper_draft" in value:

                st.write(
                    value["paper_draft"]["full_report"]
                )
