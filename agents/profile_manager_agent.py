# agents/profile_manager_agent.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

from graph_state import GraphState
from tools.profile_tools import ProfileUpdates

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
    
    # --- Part 1: Extract Profile Information ---
    extraction_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Or another capable model
    structured_extraction_llm = extraction_llm.with_structured_output(ProfileUpdates)
    
    # --- UPDATED GERMAN-CONTEXT EXTRACTION PROMPT ---
    extraction_prompt_text = """You are an expert at extracting user profile information for a German tax context.
Given the conversation history and the user's last message, extract any relevant details about their profile.
Focus on key German tax concepts like being a 'Freiberufler' (freelancer).

--- DATA TYPE & INFERENCE RULES (VERY IMPORTANT) ---
1.  `has_dependents`: This field MUST be a native JSON boolean (`true` or `false`), NOT a string. If the user mentions a 'wife', 'spouse', 'son', 'daughter', or 'child', you MUST set `has_dependents` to `true`.
2.  `marital_status`: This field MUST be a string (e.g., "married", "single"). If the user mentions a 'wife', 'spouse', or 'husband', you MUST set `marital_status` to "married". DO NOT use a boolean for this field.
3.  `known_expenses`: This MUST be a list of strings. Extract the concepts, not just the numbers (e.g., "Laptop purchase", "Train ticket").
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
        
    # --- Part 2: Decide on the Next Action ---
    print("--- Deciding next action after profile update ---")
    routing_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Or another capable model
    structured_routing_llm = routing_llm.with_structured_output(NextAction)
    
    # --- UPDATED GERMAN-CONTEXT ROUTING PROMPT ---
    routing_prompt_text = """You are an expert at analyzing user messages in a German tax context to determine if they contain a question that needs an answer.

--- RULES ---
1.  Your primary job is to find a question. Look for question words like 'how', 'what', 'why', 'when', 'is this', or a question mark '?'.
2.  If a direct question is present, you MUST choose 'knowledge_agent', even if the sentence also contains statements.
3.  Only choose 'end_conversation' if the message is PURELY a statement of fact with no follow-up question.

--- EXAMPLES ---
- User Message: "As a Freiberufler, how does the Home-Office-Pauschale work?" -> Decision: 'knowledge_agent'
- User Message: "I am married and want to know about the ELSTER portal." -> Decision: 'knowledge_agent'
- User Message: "My train ticket to the client cost 80 euros." -> Decision: 'end_conversation'
- User Message: "Just so you know, my occupation is a doctor." -> Decision: 'end_conversation'
"""
    
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", routing_prompt_text),
        ("human", "{latest_message}")
    ])
    
    routing_chain = routing_prompt | structured_routing_llm
    
    next_action_result = await routing_chain.ainvoke({"latest_message": latest_message})
    
    next_node = next_action_result.next_node
    print(f"--- Profile Manager decided next node is: {next_node} ---")

    confirmation_response = None
    if updates and next_node == "end_conversation":
        confirmation_response = f"Got it. I've updated your profile with: {', '.join(updates.keys())}."

    return {"user_profile": current_profile, "next_node": next_node, "direct_response": confirmation_response}