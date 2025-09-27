# app.py - Refactored Version

"""
Taxfix Multi-Agent Assistant - Main Streamlit Application

This refactored version includes:
- Better error handling and state validation
- Improved memory management
- Enhanced user experience
- Configuration management
- Performance monitoring
- Robust state management
"""

import streamlit as st
from dotenv import load_dotenv
import os
import re
import asyncio
import time
import hashlib
from datetime import datetime
from functools import partial
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Import core LangChain and LangGraph components
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Import the application's custom modules
from graph_state import GraphState
from rag_builder import build_retriever
from agents.knowledge_agent import create_knowledge_agent, run_knowledge_agent
from agents.router_agent import run_router
from agents.profile_manager_agent import run_profile_manager
from agents.action_proposer_agent import run_action_proposer


# --- CONFIGURATION ---
@dataclass
class AppConfig:
    """Application configuration settings."""
    MAX_MESSAGES: int = 50
    GRAPH_TIMEOUT: int = 30  # seconds
    DEBUG_MODE: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    MAX_RETRIES: int = 3
    ENABLE_PERFORMANCE_MONITORING: bool = True


# --- IMPROVED DATA STRUCTURES ---
@dataclass
class UserProfile:
    """User profile with proper defaults."""
    occupation: Optional[str] = None
    marital_status: Optional[str] = None
    has_dependents: bool = False
    is_kleinunternehmer: Optional[bool] = None
    known_expenses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "occupation": self.occupation,
            "marital_status": self.marital_status,
            "has_dependents": self.has_dependents,
            "is_kleinunternehmer": self.is_kleinunternehmer,
            "known_expenses": self.known_expenses.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from dictionary."""
        return cls(
            occupation=data.get("occupation"),
            marital_status=data.get("marital_status"),
            has_dependents=data.get("has_dependents", False),
            is_kleinunternehmer=data.get("is_kleinunternehmer"),
            known_expenses=data.get("known_expenses", []).copy()
        )


# --- UTILITY FUNCTIONS ---
def clean_response_text(text: str) -> str:
    """Cleans up common formatting artifacts from LLMs while preserving paragraph breaks."""
    if not text:
        return ""
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    return '\n'.join(cleaned_lines)


def get_api_key_hash() -> str:
    """Generate hash of API key for cache invalidation."""
    api_key = os.getenv("GROQ_API_KEY", "")
    return hashlib.md5(api_key.encode()).hexdigest()


def get_loading_message(last_action: str = None) -> str:
    """Get contextual loading message based on last action."""
    messages = {
        "router": "🧭 Analyzing your request...",
        "profile_manager": "👤 Updating your profile...",
        "knowledge_agent": "🔍 Searching tax information...",
        "action_proposer": "💡 Finding helpful suggestions...",
        "thinking": "🤔 Processing your request..."
    }
    return messages.get(last_action, "🤖 Thinking...")


def validate_session_state() -> None:
    """Ensure session state is in a valid state."""
    config = AppConfig()
    
    # Validate messages
    if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
        st.session_state.messages = [AIMessage(content="Hello! How can I help you with your taxes today?")]
    
    # Validate and clean messages
    valid_messages = []
    for msg in st.session_state.messages:
        if isinstance(msg, BaseMessage) and hasattr(msg, 'content'):
            valid_messages.append(msg)
    st.session_state.messages = valid_messages
    
    # Validate user profile
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = UserProfile().to_dict()
    elif not isinstance(st.session_state.user_profile, dict):
        st.session_state.user_profile = UserProfile().to_dict()
    
    # Initialize other state variables
    default_state = {
        "proposed_action": None,
        "run_graph_on_load": False,
        "last_graph_state": {},
        "processing_time": None,
        "error_count": 0,
        "last_error": None
    }
    
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def manage_message_history() -> None:
    """Manage message history to prevent memory issues."""
    config = AppConfig()
    if len(st.session_state.messages) > config.MAX_MESSAGES:
        # Keep first message (greeting) and last MAX_MESSAGES-1
        greeting = st.session_state.messages[0]
        recent_messages = st.session_state.messages[-(config.MAX_MESSAGES-1):]
        st.session_state.messages = [greeting] + recent_messages


# --- GRAPH BUILDING ---
@st.cache_resource
def build_graph(_api_key_hash: str):
    """
    Builds and compiles the LangGraph StateGraph.
    Cache key includes API key hash for proper invalidation.
    """
    try:
        print("--- Building graph... ---")
        
        # Initialize tools and chains
        retriever = build_retriever()
        rag_chain = create_knowledge_agent(retriever)
        knowledge_agent_node = partial(run_knowledge_agent, rag_chain=rag_chain)
        
        # Define the graph structure
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", run_router)
        workflow.add_node("profile_manager", run_profile_manager)
        workflow.add_node("knowledge_agent", knowledge_agent_node)
        workflow.add_node("action_proposer", run_action_proposer)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next_node"],
            {
                "knowledge_agent": "knowledge_agent",
                "profile_manager": "profile_manager",
                "end_conversation": END,
            },
        )
        
        workflow.add_conditional_edges(
            "profile_manager",
            lambda state: state["next_node"],
            {
                "knowledge_agent": "knowledge_agent",
                "end_conversation": "action_proposer",
            },
        )
        
        workflow.add_edge("knowledge_agent", "action_proposer")
        workflow.add_edge("action_proposer", END)
        
        compiled_graph = workflow.compile()
        print("--- Graph built successfully! ---")
        return compiled_graph
        
    except Exception as e:
        st.error(f"Failed to build graph: {str(e)}")
        raise


# --- GRAPH EXECUTION ---
async def run_graph_async(graph, graph_input: Dict[str, Any], timeout: int = 30):
    """Async wrapper to run the graph with timeout."""
    try:
        return await asyncio.wait_for(graph.ainvoke(graph_input), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Graph execution timed out after {timeout} seconds")


def run_agent() -> bool:
    """
    Execute the agent graph with comprehensive error handling.
    Returns True if successful, False if error occurred.
    """
    config = AppConfig()
    start_time = time.time()
    
    try:
        # Validate state before execution
        validate_session_state()
        
        # Prepare graph input
        graph_input = {
            "messages": st.session_state.messages,
            "user_profile": st.session_state.user_profile,
        }
        
        # Get the graph
        app = build_graph(get_api_key_hash())
        
        # Execute graph with timeout
        final_state = asyncio.run(
            run_graph_async(app, graph_input, config.GRAPH_TIMEOUT)
        )
        
        # Update session state with results
        st.session_state.user_profile = final_state.get(
            'user_profile', 
            st.session_state.user_profile
        )
        st.session_state.last_graph_state = final_state
        
        # Add direct response to messages
        direct_response = final_state.get("direct_response")
        if direct_response:
            cleaned_response = clean_response_text(direct_response)
            st.session_state.messages.append(AIMessage(content=cleaned_response))
        
        # Save proposed action
        st.session_state.proposed_action = final_state.get("proposed_action")
        
        # Performance monitoring
        processing_time = time.time() - start_time
        st.session_state.processing_time = processing_time
        
        if config.DEBUG_MODE and config.ENABLE_PERFORMANCE_MONITORING:
            st.success(f"⏱️ Processing completed in {processing_time:.2f}s")
        
        # Manage message history
        manage_message_history()
        
        # Reset error count on success
        st.session_state.error_count = 0
        st.session_state.last_error = None
        
        return True
        
    except TimeoutError as e:
        st.error("⏰ The request took too long to process. Please try a simpler question.")
        st.session_state.last_error = str(e)
        return False
        
    except Exception as e:
        st.session_state.error_count += 1
        st.session_state.last_error = str(e)
        
        error_msg = "I apologize, but I encountered an error. Please try again."
        if st.session_state.error_count > config.MAX_RETRIES:
            error_msg += " If this continues, please refresh the page."
        
        st.error(f"🚨 System Error: {str(e)}")
        st.session_state.messages.append(AIMessage(content=error_msg))
        
        if config.DEBUG_MODE:
            st.exception(e)
        
        return False


# --- UI COMPONENTS ---
def display_profile_nicely(profile: Dict[str, Any]) -> None:
    """Display user profile in a user-friendly format."""
    st.write("**👤 Your Profile**")
    
    occupation = profile.get("occupation", "Not specified")
    st.write(f"**Occupation:** {occupation}")
    
    marital_status = profile.get("marital_status", "Not specified")
    st.write(f"**Marital Status:** {marital_status}")
    
    has_dependents = profile.get("has_dependents", False)
    st.write(f"**Dependents:** {'Yes' if has_dependents else 'No'}")
    
    is_kleinunternehmer = profile.get("is_kleinunternehmer")
    if is_kleinunternehmer is not None:
        st.write(f"**Kleinunternehmer:** {'Yes' if is_kleinunternehmer else 'No'}")
    
    known_expenses = profile.get("known_expenses", [])
    if known_expenses:
        st.write("**Known Expenses:**")
        for expense in known_expenses:
            st.write(f"• {expense}")
    else:
        st.write("**Known Expenses:** None recorded yet")


def export_conversation() -> None:
    """Provide conversation export functionality."""
    if len(st.session_state.messages) <= 1:  # Only greeting message
        return
    
    conversation_text = f"Taxfix AI Assistant Conversation\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    conversation_text += "\n".join([
        f"{msg.type.title()}: {msg.content}" 
        for msg in st.session_state.messages
    ])
    
    st.download_button(
        "💾 Export Conversation",
        conversation_text,
        file_name=f"tax_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        help="Download your conversation as a text file"
    )


def reset_conversation() -> None:
    """Reset the conversation and user profile."""
    st.session_state.messages = [AIMessage(content="Hello! How can I help you with your taxes today?")]
    st.session_state.user_profile = UserProfile().to_dict()
    st.session_state.proposed_action = None
    st.session_state.last_graph_state = {}
    st.session_state.error_count = 0
    st.session_state.last_error = None
    st.rerun()


def render_action_buttons(action: Dict[str, Any]) -> None:
    """Render interactive action buttons."""
    action_id = hash(str(action))  # Create unique ID for this action
    
    with st.container(border=True):
        st.markdown(f"💡 {action.get('rationale', 'I have a suggestion:')}")
        
        col1, col2, _ = st.columns([1, 1, 3])
        
        if col1.button("✅ Add an expense", key=f"add_expense_{action_id}"):
            prompt_text = "Of course! What expenses would you like to add? Please list them (e.g., 'Office supplies 50 €, business travel 300 €')."
            st.session_state.messages.append(AIMessage(content=prompt_text))
            st.session_state.proposed_action = None
            st.rerun()
        
        if col2.button("❌ Not now", key=f"not_now_{action_id}"):
            st.session_state.messages.append(AIMessage(content="Okay, just let me know when you're ready!"))
            st.session_state.proposed_action = None
            st.rerun()


# --- MAIN APPLICATION ---
def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="Taxfix AI Assistant", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Taxfix Multi-Agent Assistant 🤖")
    
    # Initialize and validate session state
    validate_session_state()
    
    config = AppConfig()
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        st.info("This panel shows the system's internal state and controls.")
        
        # User Profile Display
        with st.expander("👤 User Profile (Long-Term Memory)", expanded=True):
            display_profile_nicely(st.session_state.user_profile)
        
        # System State (for debugging)
        if config.DEBUG_MODE:
            with st.expander("🧠 System State (Debug)", expanded=False):
                st.json({
                    "message_count": len(st.session_state.messages),
                    "has_proposed_action": st.session_state.proposed_action is not None,
                    "processing_time": st.session_state.processing_time,
                    "error_count": st.session_state.error_count
                })
        
        # Controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset", help="Start a new conversation"):
                reset_conversation()
        
        with col2:
            export_conversation()
        
        # Error information
        if st.session_state.error_count > 0:
            with st.expander("⚠️ Error Information", expanded=False):
                st.error(f"Error count: {st.session_state.error_count}")
                if st.session_state.last_error:
                    st.code(st.session_state.last_error)
    
    # Main Chat Interface
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    # Display action buttons if available
    if st.session_state.proposed_action:
        with st.chat_message("assistant"):
            render_action_buttons(st.session_state.proposed_action)
    
    # Handle new user input
    if prompt := st.chat_input("Ask about deductions or update your profile..."):
        # Add user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.proposed_action = None  # Clear any pending actions
        st.session_state.run_graph_on_load = True
        st.rerun()
    
    # Execute agent if needed
    if st.session_state.run_graph_on_load:
        st.session_state.run_graph_on_load = False
        
        with st.chat_message("assistant"):
            with st.spinner(get_loading_message("thinking")):
                success = run_agent()
                if success:
                    st.rerun()  # Only rerun on success
                # On failure, error message is already displayed


if __name__ == "__main__":
    main()