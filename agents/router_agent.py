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
            'freelancer', 'employee', 'self-employed',
            '€', 'euro', 'euros', 'per month', 'per year', 'annually', 'monthly'
        ]
        
        has_profile_info = any(indicator in message_lower for indicator in profile_indicators)
        
        # Conversation ending detection
        ending_indicators = [
            'thanks', 'thank you', 'goodbye', 'bye', 'see you',
            'that\'s all', 'that\'s it', 'no more', 'i\'m done',
            'perfect', 'great', 'ok', 'okay', 'got it', 'understood'
        ]
        
        is_ending = any(indicator in message_lower for indicator in ending_indicators) and len(last_message.split()) <= 4
        
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
   - Simple acknowledgments ("OK", "Thanks", "Got it") with no questions
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
            
            # Priority 1: Questions always go to knowledge agent
            if context_analysis['has_question']:
                logger.info("Question detected - routing to knowledge_agent")
                return {
                    "next_node": "knowledge_agent",
                    "routing_confidence": 0.9,
                    "routing_reasoning": "User asked a question - prioritizing answer over profile update"
                }
            
            # Priority 2: Conversation ending
            if context_analysis['is_ending'] and context_analysis['message_length'] <= 4:
                logger.info("Conversation ending detected")
                return {
                    "next_node": "end_conversation",
                    "routing_confidence": 0.8,
                    "routing_reasoning": "Short acknowledgment indicates conversation ending"
                }
            
            # Priority 3: New profile information
            if context_analysis['has_profile_info']:
                # But only if profile is not well-established or this is clearly new/corrected info
                if not profile_completeness['is_well_established']:
                    logger.info("New profile info for incomplete profile - routing to profile_manager")
                    return {
                        "next_node": "profile_manager",
                        "routing_confidence": 0.8,
                        "routing_reasoning": "User providing personal information to build profile"
                    }
                else:
                    # Profile is established, might be updating or asking for advice
                    logger.info("Profile info with established profile - checking for updates")
                    return {
                        "next_node": "profile_manager",
                        "routing_confidence": 0.7,
                        "routing_reasoning": "User updating existing profile information"
                    }
            
            # Default: Route to knowledge agent for established profiles
            if profile_completeness['is_well_established']:
                logger.info("Well-established profile, routing to knowledge_agent for advice")
                return {
                    "next_node": "knowledge_agent",
                    "routing_confidence": 0.6,
                    "routing_reasoning": "Profile established, user likely needs tax guidance"
                }
            
            # Fallback: Profile building for new users
            logger.info("Incomplete profile, routing to profile_manager")
            return {
                "next_node": "profile_manager",
                "routing_confidence": 0.5,
                "routing_reasoning": "Profile incomplete, gathering user information"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced routing: {e}")
            # Fallback to safe default
            return {
                "next_node": "knowledge_agent",
                "routing_confidence": 0.1,
                "routing_reasoning": f"Error occurred, defaulting to knowledge agent: {str(e)}"
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