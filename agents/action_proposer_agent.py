# agents/action_proposer_agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Optional

from graph_state import GraphState

# Define the structure of a potential action
class ProposedAction(BaseModel):
    """The structure for a proposed action to the user."""
    action_type: Literal["add_expense", "review_deductions", "none"] = Field(
        description="The type of action to propose. 'none' if no action is relevant."
    )
    rationale: Optional[str] = Field(
        description="A user-facing sentence explaining why this action is being suggested. e.g., 'I see you're a freelancer, but we haven't logged any expenses yet.'"
    )

async def run_action_proposer(state: GraphState):
    """
    Analyzes the user profile and suggests a next logical action.
    """
    print("--- Running Action Proposer ---")
    
    # We can use a fast model for this classification task
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    structured_llm = llm.with_structured_output(ProposedAction)
    
    system_prompt = """You are a proactive tax assistant. Your goal is to help the user by suggesting a logical next step.
Analyze the user's profile and decide if there is a relevant action to suggest.

--- CRITICAL RULES ---
1.  You MUST check the `known_expenses` list in the user profile.
2.  If the user is a 'freelancer' AND the `known_expenses` list is COMPLETELY EMPTY, you MUST suggest the 'add_expense' action.
3.  If the `known_expenses` list already contains one or more items, you MUST choose the 'none' action. DO NOT make a suggestion.

--- EXAMPLES ---
- Profile: {{'occupation': 'freelancer', 'known_expenses': []}} -> Decision: 'add_expense'
- Profile: {{'occupation': 'freelancer', 'known_expenses': ['home office']}} -> Decision: 'none'
- Profile: {{'occupation': 'doctor', 'known_expenses': []}} -> Decision: 'none'
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Profile: {user_profile}"), 
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Profile: {user_profile}"), 
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # The user profile is the only context this agent needs
        ("human", "User Profile: {user_profile}"), 
    ])
    
    chain = prompt | structured_llm
    
    proposed_action = await chain.ainvoke({"user_profile": state["user_profile"]})
    
    if proposed_action.action_type != "none":
        print(f"--- Proposing Action: {proposed_action.action_type} ---")
        # We will store this structured action in the final response field for the UI to use
        return {"proposed_action": proposed_action.dict()}
    else:
        print("--- No action proposed. ---")
        return {}