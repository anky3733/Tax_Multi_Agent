import streamlit as st
from dotenv import load_dotenv
import os
from functools import partial

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from graph_state import GraphState, UserProfile
from rag_builder import build_retriever
from agents.knowledge_agent import create_knowledge_agent, run_knowledge_agent
from agents.router_agent import route_messages
from agents.profile_manager_agent import run_profile_manager
from agents.action_proposer_agent import run_action_proposer

import re

def clean_response_text(text: str) -> str:
    """
    Cleans up common formatting artifacts from LLMs while preserving paragraph breaks.
    """
    if not text:
        return ""
    
    # 1. Fix the "l e t t e r s p a c i n g" bug.
    # The original was replacing a space with a space. This replaces it with nothing.
    # It now also handles multiple spaces between letters.
    text = re.sub(r'(?<=[a-zA-Z])\s+(?=[a-zA-Z])', ' ', text)
    
    # 2. Fix number spacing, e.g., "1 , 075" -> "1,075"
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    
    # 3. Preserve paragraph breaks (newlines) while cleaning up other whitespace.
    # We split the text into lines, clean each line, and then join them back.
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces/tabs within a single line into one space
        cleaned_line = re.sub(r'[ \t]{2,}', ' ', line.strip())
        cleaned_lines.append(cleaned_line)
        
    # Join the lines back together, preserving the structure
    return '\n'.join(cleaned_lines)


# Load environment variables
load_dotenv()

# --- 1. SETUP THE STATEFUL GRAPH ---

# Create the retriever and the RAG chain
# This is cached by Streamlit, so it only runs once
retriever = build_retriever()
rag_chain = create_knowledge_agent(retriever)

# Wrap the knowledge agent node to pass the rag_chain
# partial() "freezes" the rag_chain argument for the function
knowledge_agent_node = partial(run_knowledge_agent, rag_chain=rag_chain)

# --- Modify the graph definition ---
workflow = StateGraph(GraphState)

# Add nodes to the graph (this is the same)
workflow.add_node("router", route_messages)
workflow.add_node("knowledge_agent", knowledge_agent_node)
workflow.add_node("profile_manager", run_profile_manager)
workflow.add_node("action_proposer", run_action_proposer)

workflow.set_entry_point("router")

# Update the main router's conditional edges (this is the same)
workflow.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {
        "knowledge_agent": "knowledge_agent",
        "profile_manager": "profile_manager",
        "end_conversation": END,
    },
)

# The knowledge agent still ends the conversation for now
# workflow.add_edge("knowledge_agent", END)
workflow.add_edge("knowledge_agent", "action_proposer")

# The action proposer is now the end of the line
workflow.add_edge("action_proposer", END)

# --- CHANGE 1: DELETE THE OLD EDGE THAT CAUSED THE LOOP ---
# workflow.add_edge("profile_manager", "router") # DELETE OR COMMENT OUT THIS LINE

# --- CHANGE 2: ADD A NEW CONDITIONAL EDGE FROM THE PROFILE MANAGER ---
# This new edge uses the 'next_node' decision made *inside* the profile manager.
workflow.add_conditional_edges(
    "profile_manager",
    lambda state: state["next_node"],
    {
        "knowledge_agent": "knowledge_agent",
        "end_conversation": END,
    },
)

# Compile the graph
app = workflow.compile()



# --- 2. STREAMLIT UI ---
import asyncio

st.set_page_config(page_title="Taxfix AI Assistant", layout="wide")
st.title("Taxfix Multi-Agent Assistant 🤖")

# Sidebar for transparent reasoning and memory
with st.sidebar:
    st.header("System Status")
    st.info("This panel shows the system's internal state after each interaction.")

    # Display the user profile (long-term memory)
    with st.expander("👤 User Profile (Long-Term Memory)", expanded=True):
        profile_display = st.empty()

    # Display the graph's final state for debugging
    with st.expander("🧠 Graph State (Short-Term Memory)"):
        state_display = st.empty()

# Initialize session state for messages and user profile if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! How can I help you with your taxes today?")
    ]
if "user_profile" not in st.session_state:
    st.session_state.user_profile = UserProfile(
        occupation=None, marital_status=None, has_dependents=False, known_expenses=[]
    )

# Display the current user profile in the sidebar
profile_display.json(st.session_state.user_profile)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# This is our async runner function
async def run_graph(graph_input):
    # ainvoke() gives us the final state of the graph
    final_state = await app.ainvoke(graph_input)
    return final_state

if prompt := st.chat_input("Ask me about freelance taxes or deductions..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    graph_input = {
        "messages": st.session_state.messages,
        "user_profile": st.session_state.user_profile,
    }

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            final_state = asyncio.run(run_graph(graph_input))
            
            # Now we check both output fields independently
            response_generated = False

            # 1. Check for and display the direct response
            direct_response = final_state.get("direct_response")
            if direct_response:
                cleaned_response = clean_response_text(direct_response) # Apply the cleaning
                st.markdown(cleaned_response) # Display the cleaned version
                st.session_state.messages.append(AIMessage(content=cleaned_response))

            # 2. Check for and display the proposed action
            action = final_state.get("proposed_action")
            if action and action.get("action_type") != "none":
                # Use an st.container to group the rationale and buttons
                with st.container(border=True):
                    st.markdown(action.get('rationale', "I have a suggestion for you:"))
                    col1, col2, _ = st.columns(3)
                    if action.get("action_type") == "add_expense":
                        if col1.button("✅ Add an expense"):
                            st.success("Action noted! Let's add an expense.")
                        if col2.button("❌ Not now"):
                            st.info("Okay, just let me know when you're ready.")
                response_generated = True
            
            # 3. Handle the fallback case if nothing was generated
            if not response_generated:
                fallback_message = "Your profile has been updated. Is there anything else I can help with?"
                st.markdown(fallback_message)
                st.session_state.messages.append(AIMessage(content=fallback_message))


            # Update sidebar displays from the final state
            state_display.json(final_state)
            st.session_state.user_profile = final_state.get('user_profile', {})
            profile_display.json(st.session_state.user_profile)