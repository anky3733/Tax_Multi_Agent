# agents/router_agent.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_state import GraphState

# Define the structure of the routing decision
class RouteQuery(BaseModel):
    """Routes the user's query to the appropriate tool or agent."""
    next_node: Literal["knowledge_agent", "end_conversation"] = Field(
        description="The next node to route the query to. Use 'end_conversation' for simple greetings or non-tax related questions."
    )

async def route_messages(state: GraphState):
    """
    Routes the user's message to the correct agent.
    """
    print("--- Routing Message ---")
    
    # Use the Gemini Flash model for routing
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    # Bind the Pydantic model to the LLM to enable structured output
    structured_llm = llm.with_structured_output(RouteQuery)
    
    # Create a prompt to guide the LLM's routing decision
    system_prompt = """You are an expert at routing a user's request to the correct agent.
    Based on the user's last message, choose the appropriate next agent to call.
    - If the user is asking a specific question about taxes, deductions, or their financial situation, route to 'knowledge_agent'.
    - If the user is just saying hello, thank you, or having a casual conversation, route to 'end_conversation'.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{last_message}")
    ])
    
    # The chain that combines the prompt, LLM, and structured output
    router_chain = prompt | structured_llm
    
    # Get the last message from the state
    last_message = state["messages"][-1].content
    
    # Invoke the chain to get the routing decision
    result = await router_chain.ainvoke({"last_message": last_message})
    
    print(f"--- Router Decision: {result.next_node} ---")
    return {"next_node": result.next_node}