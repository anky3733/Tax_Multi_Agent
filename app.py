# main_app.py

import streamlit as st
from dotenv import load_dotenv
import os
import re
import asyncio
from functools import partial

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from graph_state import GraphState, UserProfile
from rag_builder import build_retriever
from agents.knowledge_agent import create_knowledge_agent, run_knowledge_agent
from agents.router_agent import route_messages
from agents.profile_manager_agent import run_profile_manager
from agents.action_proposer_agent import run_action_proposer
from agents.expense_handler_agent import run_expense_handler # Assuming you have this file

# --- 1. UTILITY FUNCTIONS & ENVIRONMENT SETUP ---

def clean_response_text(text: str) -> str:
    """
    Cleans up common formatting artifacts from LLMs while preserving paragraph breaks.
    """
    if not text:
        return ""

    # --- THIS IS THE CORRECTED SECTION ---
    # The previous regex that removed spaces between words has been removed.
    # This new logic correctly preserves paragraphs and cleans up extra whitespace within lines.
    
    lines = text.split('\n')
    # 1. Strip leading/trailing whitespace from each line.
    # 2. Collapse multiple spaces/tabs within a line into a single space.
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    
    # Join the cleaned lines back together, preserving the paragraph structure.
    return '\n'.join(cleaned_lines)
    # --- END OF CORRECTION ---

# Load environment variables
load_dotenv()


# --- 2. SETUP THE STATEFUL GRAPH ---

# This section is cached by Streamlit to avoid rebuilding on every interaction
@st.cache_resource
def build_graph():
    print("--- Building graph... ---")
    retriever = build_retriever()
    rag_chain = create_knowledge_agent(retriever)
    knowledge_agent_node = partial(run_knowledge_agent, rag_chain=rag_chain)
    
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", route_messages)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("profile_manager", run_profile_manager)
    workflow.add_node("action_proposer", run_action_proposer)
    workflow.add_node("expense_handler", run_expense_handler)
    
    # Define edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "knowledge_agent": "knowledge_agent",
            "profile_manager": "profile_manager",
            "expense_handler": "expense_handler",
            "end_conversation": END,
        },
    )
    
    workflow.add_conditional_edges(
        "profile_manager",
        lambda state: state["next_node"],
        {
            "knowledge_agent": "knowledge_agent",
            "end_conversation": "action_proposer", # Go to proposer even when ending
        },
    )
    
    workflow.add_edge("knowledge_agent", "action_proposer")
    workflow.add_edge("expense_handler", "action_proposer")
    workflow.add_edge("action_proposer", END)
    
    return workflow.compile()

# Build the graph once
app = build_graph()


# --- 3. AGENT EXECUTION LOGIC ---

async def run_graph_async(graph_input):
    """Asynchronously invokes the graph."""
    return await app.ainvoke(graph_input)

def run_agent():
    """
    Executes the agent graph with the current session state and updates the state.
    This function ONLY handles state mutation, not UI rendering.
    """
    graph_input = {
        "messages": st.session_state.messages,
        "user_profile": st.session_state.user_profile,
    }
    
    # Run the graph and get the final state
    final_state = asyncio.run(run_graph_async(graph_input))
    
    # --- Update Session State based on graph output ---
    
    # Update user profile
    st.session_state.user_profile = final_state.get('user_profile', st.session_state.user_profile)
    
    # Check for a direct response and add it to the chat history
    response_generated = False
    direct_response = final_state.get("direct_response")
    if direct_response:
        cleaned_response = clean_response_text(direct_response)
        st.session_state.messages.append(AIMessage(content=cleaned_response))
        response_generated = True

    # If no direct response but the conversation ended, add a fallback message
    if not response_generated and final_state.get("next_node") == "end_conversation":
        fallback_message = "Your profile has been updated. Is there anything else I can help with?"
        st.session_state.messages.append(AIMessage(content=fallback_message))

    # Save any proposed action to the session state for the UI to render
    st.session_state.proposed_action = final_state.get("proposed_action")
    
    # Store the final state for debugging in the sidebar
    st.session_state.last_graph_state = final_state


# --- 4. STREAMLIT UI RENDERING ---

def display_action_buttons():
    """
    Renders the action buttons if an action is present in the session state.
    This is the SINGLE source of truth for rendering buttons.
    """
    action = st.session_state.get("proposed_action")
    if not action or action.get("action_type") == "none":
        return

    # Display the buttons within an AI message container
    with st.chat_message("ai"):
        with st.container(border=True):
            st.markdown(action.get('rationale', "I have a suggestion:"))
            col1, col2, _ = st.columns([1, 1, 3])
            
            if col1.button("✅ Add an expense", key="add_expense_btn"):
                prompt_text = "Of course. What are the expenses you would like to add? Please list them (e.g., 'Office supplies 50 €, business travel 300 €').."
                st.session_state.messages.append(AIMessage(content=prompt_text))
                st.session_state.proposed_action = None # Clear action
                st.rerun()
            
            if col2.button("❌ Not now", key="not_now_btn"):
                st.session_state.messages.append(AIMessage(content="Okay, just let me know when you're ready."))
                st.session_state.proposed_action = None # Clear action
                st.rerun()

# --- Main App ---

st.set_page_config(page_title="Taxfix AI Assistant", layout="wide")
st.title("Taxfix Multi-Agent Assistant 🤖")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! How can I help you with your taxes today?")]
if "user_profile" not in st.session_state:
    st.session_state.user_profile = UserProfile(occupation=None, marital_status=None, has_dependents=False, known_expenses=[])
if "proposed_action" not in st.session_state:
    st.session_state.proposed_action = None
if "run_graph_on_load" not in st.session_state:
    st.session_state.run_graph_on_load = False
if "last_graph_state" not in st.session_state:
    st.session_state.last_graph_state = {}

# --- Sidebar for Transparency ---
with st.sidebar:
    st.header("System Status")
    st.info("This panel shows the system's internal state after each interaction.")
    
    with st.expander("👤 User Profile (Long-Term Memory)", expanded=True):
        st.json(st.session_state.user_profile)
    
    with st.expander("🧠 Last Graph State (Short-Term Memory)"):
        st.json(st.session_state.last_graph_state)

# --- Main Chat Interface ---

# Display all chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Display action buttons if available
display_action_buttons()

# Handle user input
if prompt := st.chat_input("Ask about deductions or update your profile..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.proposed_action = None  # Clear any old actions
    st.session_state.run_graph_on_load = True
    st.rerun()

# --- Core Agent Trigger ---
# This block runs ONLY when the flag is set by user input.
if st.session_state.run_graph_on_load:
    st.session_state.run_graph_on_load = False  # Reset the flag
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            run_agent() # Run the agent to update the state
            st.rerun() # Rerun to display the new state (AI message, buttons)