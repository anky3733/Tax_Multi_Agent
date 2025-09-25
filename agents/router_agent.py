# agents/router_agent.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from graph_state import GraphState
from langchain_groq import ChatGroq

class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate tool or agent."""
    next_node: Literal["knowledge_agent", "profile_manager","end_conversation"] = Field(
        description="The next node to route the query to..."
    )

async def route_messages(state: GraphState):
    """
    Routes the user's message to the correct agent.
    """
    print("--- Routing Message ---")
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Or another capable model
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # --- UPDATED GERMAN-CONTEXT EXAMPLES ---
    system_prompt = """You are an expert at routing a user's request to the correct agent for a German tax context.
Based on the user's last message, choose the appropriate next agent to call. Your primary goal is to first capture any new personal information.

--- RULES ---
1.  **PRIORITY 1: Profile Manager:** If the user's message contains ANY new personal details (like their job, marital status, dependents, expenses, etc.), you MUST route to 'profile_manager', even if the message also contains a question.
2.  **PRIORITY 2: Knowledge Agent:** If the message is a direct question that does NOT contain new personal details, route to 'knowledge_agent'.
3.  **PRIORITY 3: End Conversation:** If the message is a simple greeting, thank you, or off-topic, route to 'end_conversation'.

--- EXAMPLES ---
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
    
    last_message = state["messages"][-1].content
    
    result = await router_chain.ainvoke({"last_message": last_message})
    
    print(f"--- Router Decision: {result.next_node} ---")
    return {"next_node": result.next_node}