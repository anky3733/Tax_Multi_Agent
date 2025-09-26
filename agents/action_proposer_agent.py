# agents/action_proposer_agent.py

# This agent's responsibility is to be proactive. After a user's question has been answered,
# this agent analyzes the user's profile to see if there is a logical next step it can suggest,
# making the assistant more helpful and guiding.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal, Optional

from graph_state import GraphState

# We use a Pydantic model to define the *exact* structure of the LLM's output.
# This is far more reliable than parsing text. It forces the LLM to respond in a predictable JSON format.
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
    This is the entry point function for the Action Proposer when called by the LangGraph orchestrator.
    It analyzes the user profile and returns a structured action, or nothing.

    Args:
        state: The current state of the graph, containing the user profile.

    Returns:
        A dictionary with the key 'proposed_action' to update the main graph's state, or an empty dictionary.
    """
    print("--- Running Action Proposer ---")
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # A capable model is needed for reliable structured output.
    
    # This method binds our Pydantic model to the LLM, instructing it to generate a valid JSON object
    # that conforms to the `ProposedAction` schema.
    structured_llm = llm.with_structured_output(ProposedAction)
    
    # This prompt defines the agent's simple decision-making logic.
    system_prompt = """You are a proactive tax assistant for the German tax system.
Analyze the user's profile and decide if there is a relevant action to suggest.

--- CRITICAL RULES ---
# These rules are designed to make the agent's behavior predictable and prevent it from firing at the wrong time.
1.  Check the `known_expenses` list in the user profile.
2.  If the user's occupation is 'freelancer' (*Freiberufler*) AND the `known_expenses` list is COMPLETELY EMPTY, you MUST suggest the 'add_expense' action.
3.  If the `known_expenses` list already contains items, you MUST choose the 'none' action.

--- EXAMPLES ---
# Few-shot examples make the rules unambiguous for the LLM.
# Note the double curly braces {{ }} to escape them for the prompt template.
- Profile: {{'occupation': 'freelancer', 'known_expenses': []}} -> Decision: 'add_expense'
- Profile: {{'occupation': 'freelancer', 'known_expenses': ['Home Office Lump Sum']}} -> Decision: 'none'
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # This agent only needs the user's profile to make its decision, not the whole conversation history.
        ("human", "User Profile: {user_profile}"), 
    ])
    
    # Create the simple chain: feed the prompt to the structured LLM.
    chain = prompt | structured_llm
    
    # Execute the chain. The input is just the user profile from the main graph state.
    proposed_action = await chain.ainvoke({"user_profile": state["user_profile"]})
    
    # Based on the LLM's structured output, decide what to return to the graph.
    if proposed_action.action_type != "none":
        print(f"--- Proposing Action: {proposed_action.action_type} ---")
        # Return a dictionary to update the 'proposed_action' field in the `GraphState`.
        # The Streamlit UI will use this to display interactive buttons.
        return {"proposed_action": proposed_action.dict()}
    else:
        print("--- No action proposed. ---")
        # Return an empty dictionary, which tells LangGraph to make no changes to the state.
        return {}