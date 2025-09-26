# agents/router_agent.py

# This agent is the main entry point and traffic controller for the entire system.
# Its only job is to look at the user's most recent message and decide which specialized agent
# should handle it first. This separation of concerns is a core principle of multi-agent design.

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from graph_state import GraphState
from langchain_groq import ChatGroq

# A Pydantic model defines the possible outputs of the router.
# This forces the LLM to make a clean, predictable choice from a predefined list.
class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate tool or agent."""
    next_node: Literal["knowledge_agent", "profile_manager","end_conversation"] = Field(
        description="The next node to route the query to..."
    )

async def run_router(state: GraphState): # Renamed for clarity
    """
    This is the entry point function for the Router when called by the LangGraph orchestrator.
    It analyzes the last message and returns a routing decision.

    Args:
        state: The current state of the graph, which includes the full message history.

    Returns:
        A dictionary with the key 'next_node', which LangGraph uses for conditional routing.
    """
    print("--- Routing Message ---")
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # This prompt defines the router's logic. The rules are prioritized to ensure
    # that capturing user information is always the first step.
    system_prompt = """You are an expert at routing a user's request to the correct agent for a German tax context.
Based on the user's last message, choose the appropriate next agent to call. Your primary goal is to first capture any new personal information.

--- RULES ---
# By prioritizing the Profile Manager, we ensure the system's memory is always as up-to-date as possible.
1.  **PRIORITY 1: Profile Manager:** If the user's message contains ANY new personal details (like their job, marital status, dependents, expenses, etc.), you MUST route to 'profile_manager', even if the message also contains a question.
2.  **PRIORITY 2: Knowledge Agent:** If the message is a direct question that does NOT contain new personal details, route to 'knowledge_agent'.
3.  **PRIORITY 3: End Conversation:** If the message is a simple greeting, thank you, or off-topic, route to 'end_conversation'.

--- EXAMPLES ---
# Few-shot examples are critical for making the router's behavior reliable, especially for ambiguous inputs.
- User Input: "As a Freiberufler, I'm wondering how the Home-Office-Pauschale works."
- Correct Route: "profile_manager"

- User Input: "I am married and want to know about the ELSTER portal."
- Correct Route: "profile_manager"

- User Input: "What is the Kleinunternehmerregelung?"
- Correct Route: "knowledge_agent"

- User Input: "thanks that's all for now"
- Correct Route: "end_conversation"
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{last_message}")
    ])
    
    router_chain = prompt | structured_llm
    
    # The router only needs to consider the most recent message to make its decision.
    last_message = state["messages"][-1].content
    
    # Execute the chain to get the structured routing decision.
    result = await router_chain.ainvoke({"last_message": last_message})
    
    print(f"--- Router Decision: {result.next_node} ---")
    
    # Return a dictionary that updates the 'next_node' field in the `GraphState`.
    # LangGraph's conditional routing logic will use this value to decide which agent to call next.
    return {"next_node": result.next_node}

