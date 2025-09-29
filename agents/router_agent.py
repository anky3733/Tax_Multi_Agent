# agents/router_agent.py - Enhanced Version

"""
Router Agent - Main entry point and traffic controller for the tax assistance system.

Enhanced to better handle conversation context and avoid routing loops.
"""

import logging
from typing import Literal, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from graph_state import GraphState

# Configure logging for better debugging and monitoring
logger = logging.getLogger(__name__)


class RouteQuery(BaseModel):
    """
    Structured output model for routing decisions.
    """
    next_node: Literal["knowledge_agent", "profile_manager", "end_conversation"] = Field(
        description="The next node to route the query to based on user intent and content"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the routing decision (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why this routing decision was made"
    )


class RouterAgent:
    """
    Enhanced Router Agent with better context awareness and routing logic.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._structured_llm = None
    
    @property
    def llm(self):
        """Lazy initialization of the LLM to avoid unnecessary API calls."""
        if self._llm is None:
            try:
                self._llm = ChatGroq(model=self.model_name, temperature=self.temperature)
                self._structured_llm = self._llm.with_structured_output(RouteQuery)
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise
        return self._structured_llm
    
    def _analyze_message_context(self, last_message: str, user_profile: dict, recent_messages: List[str]) -> dict:
        """
        Analyze message context to provide better routing hints.
        
        Returns:
            Dictionary with context analysis
        """
        message_lower = last_message.lower()
        
        # Question detection (high priority for knowledge_agent)
        question_indicators = [
            '?', 'what', 'how', 'why', 'when', 'where', 'which', 'who',
            'should i', 'can i', 'could i', 'would i', 'may i',
            'tell me', 'explain', 'help me', 'advice', 'recommend', 'suggest',
            'what about', 'how about', 'what if', 'is it', 'do i', 'does it'
        ]
        
        has_question = any(indicator in message_lower for indicator in question_indicators)
        
        # Profile information detection
        profile_indicators = [
            'i am', 'i work', 'i earn', 'my income', 'my salary', 'i make',
            'i have', 'i bought', 'i purchased', 'i spent',
            'my wife', 'my husband', 'my spouse', 'my children', 'my kids',
            'married', 'single', 'divorced', 'widowed',
            'freelancer', 'freiberufler', 'employee', 'self-employed',  # Add 'freiberufler'
            '€', 'euro', 'euros', 'per month', 'per year', 'annually', 'monthly',
            'expecting to earn', 'planning to earn', 'will earn'  # Add these patterns
        ]
        
        has_profile_info = any(indicator in message_lower for indicator in profile_indicators)
        
        # Conversation ending detection
        ending_indicators = [
            'thanks', 'thank you', 'goodbye', 'bye', 'see you',
            'that\'s all', 'that\'s it', 'no more', 'i\'m done',
            'perfect', 'great', 'ok', 'okay', 'got it', 'understood'
        ]
        
        is_ending = any(indicator in message_lower for indicator in ending_indicators) and len(last_message.split()) <= 10
        
        # Check for repeated information
        profile_completeness = self._assess_profile_completeness(user_profile)
        
        return {
            'has_question': has_question,
            'has_profile_info': has_profile_info,
            'is_ending': is_ending,
            'profile_completeness': profile_completeness,
            'message_length': len(last_message.split())
        }
    
    def _assess_profile_completeness(self, user_profile: dict) -> dict:
        """Assess how complete the user profile is."""
        required_fields = ['occupation', 'marital_status', 'monthly_income']
        optional_fields = ['has_dependents', 'known_expenses']
        
        required_complete = sum(1 for field in required_fields if user_profile.get(field))
        optional_complete = sum(1 for field in optional_fields if user_profile.get(field))
        
        return {
            'required_completion': required_complete / len(required_fields),
            'optional_completion': optional_complete / len(optional_fields),
            'is_well_established': required_complete >= 2  # At least 2/3 required fields
        }
    
    def _build_enhanced_prompt(self) -> ChatPromptTemplate:
        """
        Build enhanced routing prompt with better context handling.
        """
        system_prompt = """You are an expert routing agent for a German tax assistance system.
Analyze the user's message and conversation context to make optimal routing decisions.

ENHANCED ROUTING RULES (Priority Order):

1. **KNOWLEDGE_AGENT** - Route here if:
   - User asks ANY question (explicit or implicit)
   - Message contains question words or question marks
   - User requests explanations, advice, or information
   - User mentions tax concepts they want to understand
   - User says things like "what should I do?", "help me with...", "tell me about..."
   - User has established profile and is asking for advice

2. **PROFILE_MANAGER** - Route here if:
   - User provides NEW personal/financial information
   - User corrects or updates existing profile information
   - User mentions life changes (job, marriage, income change)
   - Message contains specific numbers, amounts, or personal details
   - Profile is incomplete and user is providing missing information

3. **END_CONVERSATION** - Route here if:
   # --- START FIX 1.2: ADD MORE EXPLICIT EXAMPLES ---
   - Simple acknowledgments ("OK", "Thanks", "Got it", "Perfect, thank you") with no questions
   - The user message is very short (< 4 words) and contains a closing phrase.
   # --- END FIX 1.2 ---
   - Clear conversation ending signals
   - Pure greetings without tax context
   - User indicates they're done or satisfied

CONTEXT CONSIDERATIONS:

- **Profile Completeness**: {profile_completeness}
- **Message Analysis**: {context_analysis}
- **Recent Context**: Consider if user just provided information or asked questions

DECISION LOGIC:
- Questions ALWAYS go to knowledge_agent (even if they contain personal info)
- New personal info goes to profile_manager UNLESS it's part of a question
- Short responses (<5 words) that are just acknowledgments go to end_conversation

EXAMPLES:

"I earn 3000€ monthly as a graphic designer" → profile_manager (new personal info)
"I earn 3000€ monthly. What deductions can I claim?" → knowledge_agent (question with context)
"What home office deductions apply to freelancers?" → knowledge_agent (question)
"As a married freelancer, what should I know?" → knowledge_agent (question with context)
"I'm actually married, not single" → profile_manager (correction)
"Thanks, that helps!" → end_conversation (acknowledgment)
"OK" → end_conversation (simple acknowledgment)
# --- START FIX 1.2: ADD MORE EXPLICIT EXAMPLES ---
"Great, thank you for the help!" → end_conversation (clear closing statement)
"Perfect" → end_conversation (short acknowledgment)
# --- END FIX 1.2 ---

Always prioritize helping users get answers to their questions."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """User message: {last_message}

Context Analysis: {context_analysis}
Profile Completeness: {profile_completeness}
User Profile: {user_profile}

Route this message to the appropriate agent.""")
        ])
    
    async def route_message(self, state: GraphState) -> dict:
        """
        Enhanced routing with better context analysis.
        """
        logger.info("--- Starting enhanced message routing ---")
        
        try:
            # Validate state
            if not state.get("messages") or len(state["messages"]) == 0:
                logger.warning("No messages found in state")
                return {
                    "next_node": "end_conversation",
                    "routing_confidence": 0.5,
                    "routing_reasoning": "No messages to process"
                }
            
            # Get context
            last_message = state["messages"][-1].content.strip()
            message_lower = last_message.lower()
            simple_closings = [
                'thanks', 'thank you', 'ok', 'okay', 'got it', 'perfect', 
                'great', 'sounds good', 'bye', 'goodbye', 'that helps'
            ]
            # If the message is short and is a known closing phrase, end immediately.
            if message_lower in simple_closings and len(last_message.split()) <= 3:
                logger.info("Simple closing detected - routing to end_conversation")
                return {
                    "next_node": "end_conversation",
                    "routing_confidence": 1.0,
                    "routing_reasoning": "Detected a simple closing phrase."
                }
            user_profile = state.get("user_profile", {})
            
            # Get recent message history for context
            recent_messages = []
            for msg in state["messages"][-3:]:  # Last 3 messages for context
                recent_messages.append(msg.content)
            
            if not last_message:
                logger.warning("Empty message content")
                return {
                    "next_node": "end_conversation", 
                    "routing_confidence": 0.5,
                    "routing_reasoning": "Empty message content"
                }
            
            # Analyze message context
            context_analysis = self._analyze_message_context(last_message, user_profile, recent_messages)
            profile_completeness = context_analysis['profile_completeness']
            
            logger.info(f"Context analysis: {context_analysis}")
            
            # ENHANCED DECISION LOGIC

            if context_analysis['has_question'] and context_analysis['has_profile_info']:
                logger.info("Message contains both profile info and a question.")
                
                # This is the condition you wanted to keep.
                if not profile_completeness['is_well_established']:
                    logger.info("Profile is incomplete. Routing to 'profile_manager' to build profile first.")
                    return {
                        "next_node": "profile_manager",
                        "routing_confidence": 0.95,
                        "routing_reasoning": "User is new; capturing profile info is the priority before answering."
                    }
                else:
                    # --- THIS IS THE KEY FIX ---
                    # The user is established, but we STILL must go to the profile manager
                    # to capture the new information (e.g., the laptop purchase).
                    logger.info("Profile is established, but new info detected. Routing to 'profile_manager' to capture update before answering.")
                    return {
                        "next_node": "profile_manager", # <-- CHANGED FROM 'knowledge_agent'
                        "routing_confidence": 0.9,
                        "routing_reasoning": "Capturing new profile information from an established user before routing to answer the question."
                    }

            # PRIORITY 2: Handle pure questions (no new profile info).
            # This logic now only runs if the message is ONLY a question.
            if context_analysis['has_question']:
                logger.info("Pure question detected. Routing directly to 'knowledge_agent'.")
                return {
                    "next_node": "knowledge_agent",
                    "routing_confidence": 0.9,
                    "routing_reasoning": "User asked a question without providing new profile data."
                }
            
            # PRIORITY 3: Handle pure profile updates (no question).
            if context_analysis['has_profile_info']:
                logger.info("Pure profile update detected. Routing to 'profile_manager'.")
                return {
                    "next_node": "profile_manager",
                    "routing_confidence": 0.8,
                    "routing_reasoning": "User is providing or updating profile information."
                }
            
            # PRIORITY 4: Handle general conversation endings.
            if context_analysis['is_ending']:
                logger.info("General conversation ending detected. Routing to 'end_conversation'.")
                return {
                    "next_node": "end_conversation",
                    "routing_confidence": 0.8,
                    "routing_reasoning": "Message indicates the user is finished."
                }

            # FALLBACK for any other ambiguous input.
            logger.info("Uncertain intent. Defaulting to 'profile_manager' for final analysis.")
            return {
                "next_node": "profile_manager",
                "routing_confidence": 0.5,
                "routing_reasoning": "Fallback for ambiguous input; letting profile manager check for latent info."
            }
            
        except Exception as e:
            logger.error(f"Error in router agent: {e}", exc_info=True)
            # Safe fallback
            return {
                "next_node": "knowledge_agent",
                "routing_confidence": 0.1,
                "routing_reasoning": f"An unexpected error occurred, defaulting to knowledge agent: {str(e)}"
            }


# Global router instance
_router_instance = RouterAgent()


async def run_router(state: GraphState) -> dict:
    """
    Main entry point for the enhanced router agent.
    """
    return await _router_instance.route_message(state)


def get_router_instance() -> RouterAgent:
    """Get the global router instance for direct access."""
    return _router_instance