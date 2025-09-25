# agents/profile_manager_agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

from graph_state import GraphState
from tools.profile_tools import ProfileUpdates

# --- NEW: A Pydantic model for the internal routing decision ---
class NextAction(BaseModel):
    """Determines the next agent to call after updating the profile."""
    next_node: Literal["knowledge_agent", "end_conversation"] = Field(
        description="The next node to route to. 'knowledge_agent' if there is a question to answer, otherwise 'end_conversation'."
    )

async def run_profile_manager(state: GraphState):
    """
    Extracts profile info and then decides the next step.
    """
    print("--- Running Profile Manager ---")
    
    # --- Part 1: Extract Profile Information (existing logic) ---
    extraction_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_extraction_llm = extraction_llm.with_structured_output(ProfileUpdates)
    
    extraction_prompt_text = """You are an expert at extracting user profile information from a conversation.
    Given the conversation history and the user's last message, extract any relevant details about their profile.
    Only extract information that is explicitly mentioned.

    --- DATA TYPE & INFERENCE RULES (VERY IMPORTANT) ---
    1.  `has_dependents`: This field MUST be a native JSON boolean (`true` or `false`), NOT a string. If the user mentions a 'wife', 'spouse', 'son', 'daughter', or 'child', you MUST set `has_dependents` to `true`.
    2.  `marital_status`: This field MUST be a string (e.g., "married", "single"). If the user mentions a 'wife', 'spouse', or 'husband', you MUST set `marital_status` to "married". DO NOT use a boolean for this field.
    3.  `known_expenses`: This MUST be a list of strings.
    4.  If you do not find a value for a field, OMIT the field entirely from your output. DO NOT use null.
    """
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", extraction_prompt_text),
        ("human", "Here is the conversation history:\n<history>{history}</history>"),
        ("human", "And here is the user's latest message:\n<latest_message>{latest_message}</latest_message>"),
    ])
    
    extraction_chain = extraction_prompt | structured_extraction_llm
    
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"][:-1]])
    latest_message = state["messages"][-1].content
    
    extracted_data: ProfileUpdates = await extraction_chain.ainvoke({
        "history": history, 
        "latest_message": latest_message
    })
    
    current_profile = state["user_profile"].copy()
    updates = extracted_data.dict(exclude_unset=True)
    
    if updates:
        print(f"--- Found Profile Updates: {updates} ---")
        current_profile.update(updates)
    else:
        print("--- No new profile information found. ---")
        
    # --- Part 2: Decide on the Next Action (NEW logic) ---
    print("--- Deciding next action after profile update ---")
    routing_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0) # Use a fast model for this simple task
    structured_routing_llm = routing_llm.with_structured_output(NextAction)
    
    routing_prompt_text = """Based on the user's last message, decide if it contains a follow-up question that needs answering.
- A 'question' explicitly asks for information, often starting with 'What', 'How', 'Why', or ending with a question mark.
- An 'informational statement' simply provides data without asking for an explanation.

--- RULES ---
1. If the message is a clear question, choose 'knowledge_agent'.
2. If the message is ONLY an informational statement (like stating an expense or personal fact), choose 'end_conversation'.

--- EXAMPLES ---
- User Message: "As a freelancer, how does the home office deduction work?" -> Decision: 'knowledge_agent'
- User Message: "What is the mileage rate?" -> Decision: 'knowledge_agent'
- User Message: "Just so you know, I am married now." -> Decision: 'end_conversation'
- User Message: "My flight to the conference cost 300 dollars." -> Decision: 'end_conversation'
- User Message: "Car : 20,000 USD" -> Decision: 'end_conversation'
"""
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", routing_prompt_text),
        ("human", "{latest_message}")
    ])
    
    routing_chain = routing_prompt | structured_routing_llm
    
    next_action_result = await routing_chain.ainvoke({"latest_message": latest_message})
    
    next_node = next_action_result.next_node
    print(f"--- Profile Manager decided next node is: {next_node} ---")

    # If the profile was updated AND we are ending the conversation, add a confirmation message.
    confirmation_response = None
    if updates and next_node == "end_conversation":
        confirmation_response = f"Got it. I've updated your profile with: {', '.join(updates.keys())}."

    return {"user_profile": current_profile, "next_node": next_node, "direct_response": confirmation_response}