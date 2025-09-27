# app.py

# This is the main entry point for the Streamlit web application.
# It orchestrates the entire system:
# 1. Sets up the environment and the multi-agent graph.
# 2. Manages the application's state (chat history, user profile) using Streamlit's session_state.
# 3. Renders the user interface, including the chat messages and interactive elements.
# 4. Triggers the agent execution in response to user input with STREAMING RESPONSES.

import streamlit as st
from dotenv import load_dotenv
import os
import re
import asyncio
import time
from functools import partial
from typing import Dict, Any, AsyncGenerator, Optional

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

def simulate_streaming_text(text: str, delay: float = 0.02) -> AsyncGenerator[str, None]:
    """
    Simulates streaming text by yielding chunks of text with delays.
    This creates the typewriter effect for better UX.
    """
    async def _stream():
        words = text.split()
        current_text = ""
        
        for i, word in enumerate(words):
            current_text += word + " "
            yield current_text.strip()
            await asyncio.sleep(delay)
    
    return _stream()

def stream_text_to_container(container, text: str, delay: float = 0.02):
    """
    Streams text to a Streamlit container with typewriter effect.
    """
    placeholder = container.empty()
    words = text.split()
    current_text = ""
    
    for i, word in enumerate(words):
        current_text += word + " "
        placeholder.markdown(current_text.strip())
        time.sleep(delay)
    
    return current_text.strip()

# Load environment variables (e.g., GROQ_API_KEY) from the .env file.
load_dotenv()

# --- 2. STREAMING GRAPH EXECUTION ---

class StreamingGraphRunner:
    """
    A class to handle streaming execution of the multi-agent graph.
    This provides real-time feedback about which agent is running and what they're doing.
    """
    
    def __init__(self, graph):
        self.graph = graph
        
    async def run_with_streaming_updates(self, graph_input: Dict[str, Any], 
                                       status_container, 
                                       response_container) -> Dict[str, Any]:
        """
        Runs the graph with streaming status updates and response streaming.
        """
        # Initialize status display
        status_placeholder = status_container.empty()
        response_placeholder = response_container.empty()
        
        # Track the execution flow
        execution_steps = []
        
        try:
            # Show initial processing status
            status_placeholder.info("🤖 **Router Agent**: Analyzing your request...")
            await asyncio.sleep(0.5)  # Small delay for visual effect
            
            # Run the graph (this could be made truly streaming with custom graph execution)
            final_state = await self.graph.ainvoke(graph_input)
            
            # Simulate agent execution steps for demonstration
            # In a real implementation, you'd modify your agents to yield intermediate results
            await self._simulate_agent_execution_flow(status_placeholder, final_state)
            
            # Stream the final response
            direct_response = final_state.get("direct_response")
            if direct_response:
                status_placeholder.success("✅ **Response Ready**: Streaming answer...")
                cleaned_response = clean_response_text(direct_response)
                
                # Stream the response word by word
                stream_text_to_container(response_placeholder, cleaned_response, delay=0.03)
                
                # Clear status after response is complete
                await asyncio.sleep(0.5)
                status_placeholder.empty()
            
            return final_state
            
        except Exception as e:
            status_placeholder.error(f"❌ **Error**: {str(e)}")
            raise
    
    async def _simulate_agent_execution_flow(self, status_placeholder, final_state: Dict[str, Any]):
        """
        Simulates the agent execution flow with status updates.
        In a real implementation, you'd modify your agents to yield these updates.
        """
        # Determine which agents likely ran based on the final state
        steps = []
        
        # Router always runs first
        steps.append("🤖 **Router Agent**: Analyzing your request...")
        
        # Check if profile was updated
        if final_state.get("user_profile"):
            steps.append("👤 **Profile Manager**: Updating your profile...")
        
        # Knowledge agent likely ran if there's a direct response
        if final_state.get("direct_response"):
            steps.append("📚 **Knowledge Agent**: Searching tax database...")
            steps.append("🧠 **Knowledge Agent**: Formulating personalized answer...")
        
        # Action proposer runs if there's a proposed action
        if final_state.get("proposed_action"):
            steps.append("💡 **Action Proposer**: Analyzing potential next steps...")
        
        # Show each step with delays
        for step in steps:
            status_placeholder.info(step)
            await asyncio.sleep(0.8)  # Realistic processing time simulation


# --- 3. SETUP THE STATEFUL MULTI-AGENT GRAPH ---

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
    knowledge_agent_node = partial(run_knowledge_agent, rag_chain=rag_chain)
    
    # 2. Define the graph structure using LangGraph's StateGraph
    workflow = StateGraph(GraphState)
    
    # 3. Add the agents as nodes in the graph
    workflow.add_node("router", run_router)
    workflow.add_node("profile_manager", run_profile_manager)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("action_proposer", run_action_proposer)
    
    # 4. Define the edges
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
    
    print("--- Graph built successfully! ---")
    return workflow.compile()

# Build the graph and streaming runner
app = build_graph()
streaming_runner = StreamingGraphRunner(app)


# --- 4. ENHANCED AGENT EXECUTION WITH STREAMING ---

async def run_graph_with_streaming(status_container, response_container):
    """
    Runs the agent graph with streaming updates and response streaming.
    """
    graph_input = {
        "messages": st.session_state.messages,
        "user_profile": st.session_state.user_profile,
        "last_proposed_action_type": st.session_state.last_proposed_action_type
    }
    
    final_state = await streaming_runner.run_with_streaming_updates(
        graph_input, status_container, response_container
    )
    
    # Update session state
    st.session_state.user_profile = final_state.get('user_profile', st.session_state.user_profile)
    st.session_state.last_graph_state = final_state
    
    # Handle the streamed response (already displayed, but add to message history)
    direct_response = final_state.get("direct_response")
    if direct_response:
        cleaned_response = clean_response_text(direct_response)
        st.session_state.messages.append(AIMessage(content=cleaned_response))

    # Save proposed action
    st.session_state.proposed_action = final_state.get("proposed_action")

def run_agent_with_streaming():
    """
    Executes the agent with streaming UI updates.
    This creates containers for status and response, then runs the streaming execution.
    """
    # Create the AI message container
    with st.chat_message("ai"):
        # Create containers for streaming updates
        status_container = st.container()
        response_container = st.container()
        
        # Run the async streaming execution
        asyncio.run(run_graph_with_streaming(status_container, response_container))


# --- 5. STREAMLIT UI RENDERING ---

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
if "last_proposed_action_type" not in st.session_state:
    st.session_state.last_proposed_action_type = None

# --- Enhanced Sidebar with Streaming Status ---
with st.sidebar:
    st.header("System Status")
    st.info("This panel shows the system's internal state and processing status.")
    
    # Add streaming status indicator
    if st.session_state.run_graph_on_load:
        st.warning("🔄 **Processing your request...**")
    else:
        st.success("✅ **Ready for input**")
    
    with st.expander("👤 User Profile (Long-Term Memory)", expanded=True):
        st.json(st.session_state.user_profile)
    
    with st.expander("🧠 Last Graph State (Short-Term Memory)"):
        st.json(st.session_state.last_graph_state)
    
    # Add performance metrics
    with st.expander("📊 Session Metrics"):
        st.metric("Total Messages", len(st.session_state.messages))
        if st.session_state.user_profile:
            profile_completeness = sum([
                bool(st.session_state.user_profile.get('occupation')),
                bool(st.session_state.user_profile.get('marital_status')),
                bool(st.session_state.user_profile.get('known_expenses'))
            ]) / 3 * 100
            st.metric("Profile Completeness", f"{profile_completeness:.0f}%")

# --- Main Chat Interface ---

# Display all past messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Display proposed actions with enhanced styling

if st.session_state.proposed_action:
    action = st.session_state.proposed_action
    st.session_state.last_proposed_action_type = action.get('action_type')
    with st.chat_message("ai"):
        st.markdown("### 💡 **Suggested Action**")
        with st.container(border=True):
            st.markdown(action.get('rationale', "I have a suggestion:"))
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            # Get action type with fallback
            action_type = action.get('action_type', 'add_expense')
            
            # Define action configurations
            action_configs = {
                'add_income': {
                    'button_text': "💰 **Add Income Info**",
                    'prompt': "Great! Please tell me about your income. For example: 'I earn 3500€ per month' or 'My annual income is about 50000€'",
                },
                'update_income': {
                    'button_text': "💰 **Update Income**", 
                    'prompt': "Perfect! Please share your updated income information.",
                },
                'add_expense': {
                    'button_text': "📝 **Add Expenses**", 
                    'prompt': "Perfect! What expenses would you like to add? Format: 'Office supplies 50€, business travel 300€'",
                },
                'add_dependent': {
                    'button_text': "👶 **Add Family Info**",
                    'prompt': "I'd be happy to help with family information. Tell me about your dependents or family situation.",
                },
                'check_tax_class': {
                    'button_text': "📊 **Check Tax Class**",
                    'prompt': "Let me help you with tax class optimization. What's your current tax class situation?",
                },
                'review_deductions': {
                    'button_text': "🔍 **Review Deductions**",
                    'prompt': "Great! Let's review your deductions. What specific deductions would you like to explore?",
                }
            }
            
            # Get config with fallback
            config = action_configs.get(action_type, action_configs['add_expense'])
            
            with col1:
                if st.button(config['button_text'], key=f"{action_type}_btn", type="primary"):
                    st.session_state.messages.append(AIMessage(content=config['prompt']))
                    st.session_state.proposed_action = None
                    st.rerun()
            
            with col2:
                if st.button("❌ **Not now**", key="not_now_btn"):
                    responses = [
                        "No problem! I'm here whenever you need help.",
                        "Understood. Feel free to ask me anything else!"
                    ]
                    import random
                    response = random.choice(responses)
                    st.session_state.messages.append(AIMessage(content=response))
                    st.session_state.proposed_action = None
                    st.rerun()

# Handle new user input
if prompt := st.chat_input("Ask about deductions, update your profile, or get personalized tax advice..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.proposed_action = None
    st.session_state.last_proposed_action_type = None
    st.session_state.run_graph_on_load = True
    st.rerun()

# --- Enhanced Agent Execution with Streaming ---
if st.session_state.run_graph_on_load:
    st.session_state.run_graph_on_load = False
    
    # Run the agent with streaming updates
    run_agent_with_streaming()
    st.rerun()  # Final rerun to update the complete UI state