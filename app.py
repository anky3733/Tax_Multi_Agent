# app.py

# This is the main entry point for the Streamlit web application.
# It orchestrates the entire system:
# 1. Sets up the environment and the multi-agent graph.
# 2. Manages the application's state (chat history, user profile) using Streamlit's session_state.
# 3. Renders the user interface, including the chat messages and interactive elements.
# 4. Triggers the agent execution in response to user input.

import streamlit as st
from dotenv import load_dotenv
import os
import re
import asyncio
from functools import partial

# Import core LangChain and LangGraph components
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# Import the application's custom modules
from graph_state import GraphState, UserProfile
from rag_builder import build_retriever
from agents.knowledge_agent import create_knowledge_agent, run_knowledge_agent
from agents.router_agent import run_router # Renamed for clarity
from agents.profile_manager_agent import run_profile_manager
from agents.action_proposer_agent import run_action_proposer

# --- 1. UTILITY FUNCTIONS & ENVIRONMENT SETUP ---

def clean_response_text(text: str) -> str:
    """
    Cleans up common formatting artifacts from LLMs while preserving paragraph breaks.
    This is a crucial UX function to prevent "walls of text" or garbled output.
    """
    if not text:
        return ""
    # This logic splits the text by newlines, cleans each line individually by removing
    # excess whitespace, and then rejoins them, preserving the paragraph structure.
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    return '\n'.join(cleaned_lines)

# Load environment variables (e.g., GROQ_API_KEY) from the .env file.
load_dotenv()


# --- 2. SETUP THE STATEFUL MULTI-AGENT GRAPH ---

# The @st.cache_resource decorator is a powerful Streamlit feature.
# It ensures that this complex and slow-to-create graph object is built ONLY ONCE
# when the app starts, and then cached for all subsequent user sessions and reruns.
@st.cache_resource
def build_graph():
    """
    Builds and compiles the LangGraph StateGraph that defines the agent workflow.
    This function is cached to prevent rebuilding on every interaction.
    """
    print("--- Building graph... ---")
    
    # 1. Initialize Tools and Chains that agents will use
    retriever = build_retriever()
    rag_chain = create_knowledge_agent(retriever)
    
    # Use functools.partial to "freeze" the rag_chain argument for the knowledge agent node.
    # This is a clean way to pass dependencies into graph nodes.
    knowledge_agent_node = partial(run_knowledge_agent, rag_chain=rag_chain)
    
    # 2. Define the graph structure using LangGraph's StateGraph
    workflow = StateGraph(GraphState)
    
    # 3. Add the agents as nodes in the graph
    workflow.add_node("router", run_router)
    workflow.add_node("profile_manager", run_profile_manager)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("action_proposer", run_action_proposer)
    
    # 4. Define the edges that connect the nodes, creating the conversational flow
    workflow.set_entry_point("router")
    
    # The main router uses conditional routing based on the 'next_node' field in the state.
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_node"],
        {
            "knowledge_agent": "knowledge_agent",
            "profile_manager": "profile_manager",
            "end_conversation": END, # The END node stops the graph execution for this turn.
        },
    )
    
    # The Profile Manager also has its own internal routing logic.
    workflow.add_conditional_edges(
        "profile_manager",
        lambda state: state["next_node"],
        {
            "knowledge_agent": "knowledge_agent",
            "end_conversation": END, # If only updating the profile, the conversation can end here.
        },
    )
    
    # After the Knowledge Agent runs, we always check if a proactive action can be suggested.
    workflow.add_edge("knowledge_agent", "action_proposer")
    # The Action Proposer is the final step before the graph finishes.
    workflow.add_edge("action_proposer", END)
    
    # 5. Compile the graph into a runnable LangChain object.
    print("--- Graph built successfully! ---")
    return workflow.compile()

# Build the graph when the application starts.
app = build_graph()


# --- 3. AGENT EXECUTION LOGIC ---

async def run_graph_async(graph_input):
    """A simple async wrapper to run the graph."""
    return await app.ainvoke(graph_input)

def run_agent():
    """
    This is the core function that executes the agent graph.
    Its only job is to take the current session state, run the graph,
    and update the session state with the results. It does NOT render any UI.
    """
    graph_input = {
        "messages": st.session_state.messages,
        "user_profile": st.session_state.user_profile,
    }
    
    final_state = asyncio.run(run_graph_async(graph_input))
    
    # --- Update the session state based on the final output of the graph ---
    st.session_state.user_profile = final_state.get('user_profile', st.session_state.user_profile)
    st.session_state.last_graph_state = final_state # For debugging in the sidebar
    
    # Add the direct response (if any) to the message history
    direct_response = final_state.get("direct_response")
    if direct_response:
        cleaned_response = clean_response_text(direct_response)
        st.session_state.messages.append(AIMessage(content=cleaned_response))

    # Save the proposed action (if any) to the session state for the UI to render separately.
    st.session_state.proposed_action = final_state.get("proposed_action")


# --- 4. STREAMLIT UI RENDERING ---

# Set the page configuration for the web app.
st.set_page_config(page_title="Taxfix AI Assistant", layout="wide")
st.title("Taxfix Multi-Agent Assistant 🤖")

# --- Initialize Session State ---
# Streamlit's session_state is a dictionary that persists across script reruns.
# It's our primary tool for managing the application's state.
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
# This section provides a view into the agent's "mind".
with st.sidebar:
    st.header("System Status")
    st.info("This panel shows the system's internal state after each interaction.")
    
    with st.expander("👤 User Profile (Long-Term Memory)", expanded=True):
        st.json(st.session_state.user_profile)
    
    with st.expander("🧠 Last Graph State (Short-Term Memory)"):
        st.json(st.session_state.last_graph_state)

# --- Main Chat Interface ---

# This is the main render loop. On every rerun, it draws the UI based on the current session_state.
# 1. Display all past messages.
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# 2. Display the action buttons if an action is currently proposed.
if st.session_state.proposed_action:
    action = st.session_state.proposed_action
    with st.chat_message("ai"):
        with st.container(border=True):
            st.markdown(action.get('rationale', "I have a suggestion:"))
            col1, col2, _ = st.columns([1, 1, 3])
            
            if col1.button("✅ Add an expense", key="add_expense_btn"):
                prompt_text = "Of course. What are the expenses you would like to add? Please list them (e.g., 'Office supplies 50 €, business travel 300 €')."
                st.session_state.messages.append(AIMessage(content=prompt_text))
                st.session_state.proposed_action = None # Clear the action after handling it
                st.rerun() # Rerun to display the new prompt immediately
            
            if col2.button("❌ Not now", key="not_now_btn"):
                st.session_state.messages.append(AIMessage(content="Okay, just let me know when you're ready."))
                st.session_state.proposed_action = None # Clear the action
                st.rerun()

# 3. Handle new user input from the chat box.
if prompt := st.chat_input("Ask about deductions or update your profile..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.proposed_action = None  # User typing overrides any previous action
    st.session_state.run_graph_on_load = True # Set the flag to trigger the agent
    st.rerun() # Rerun immediately to show the user's message

# --- Core Agent Trigger ---
# This block is the heart of the interaction logic. It runs the agent ONLY when the flag is set.
if st.session_state.run_graph_on_load:
    st.session_state.run_graph_on_load = False  # Reset the flag to prevent re-running
    
    # We display the spinner in the main UI thread before running the agent
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            run_agent() # This function runs the graph and updates the state
            st.rerun()  # Rerun one last time to display the agent's new messages and actions