# agents/profile_manager_agent.py - Refactored Version

"""
Profile Manager Agent - Refactored

This agent is responsible for:
1. Extracting user profile information from conversations
2. Updating the user profile with new information
3. Deciding the next routing step based on conversation context
4. Providing confirmation responses for profile updates

Improvements:
- Better error handling and validation
- Enhanced extraction logic with confidence scoring
- Improved German tax context understanding
- More robust routing decisions
- Comprehensive logging and debugging
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from graph_state import GraphState


# --- CONFIGURATION ---
@dataclass
class ProfileManagerConfig:
    """Configuration for the Profile Manager Agent."""
    model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.0
    max_retries: int = 3
    confidence_threshold: float = 0.7
    enable_debug_logging: bool = True


# --- PYDANTIC MODELS FOR STRUCTURED OUTPUT ---
class ProfileUpdates(BaseModel):
    """Enhanced structure for profile updates with confidence scoring."""
    
    # Core profile fields
    occupation: Optional[str] = Field(
        None, 
        description="User's occupation in German tax context (e.g., 'Freiberufler', 'Angestellter', 'Beamter')"
    )
    marital_status: Optional[str] = Field(
        None, 
        description="Marital status as string: 'single', 'married', 'divorced', 'widowed'"
    )
    spouse_annual_income: Optional[float] = Field( # <-- ADD THIS
        None,
        description="Spouse's annual income in euros. Set to 0 if explicitly mentioned they don't work."
    )

    has_dependents: Optional[bool] = Field(
        None, 
        description="Boolean: true if user has children or dependents, false otherwise"
    )
    is_kleinunternehmer: Optional[bool] = Field(
        None, 
        description="Boolean: true if user is a Kleinunternehmer (small business for VAT), false otherwise"
    )
    known_expenses: Optional[List[str]] = Field(
        None, 
        description="List of expense categories or specific expenses mentioned by user"
    )

    monthly_income: Optional[int] = Field(
        None, 
        description="User's monthly income in euros"
    )
    annual_income: Optional[int] = Field(
        None, 
        description="User's annual income in euros"
    )
    
    # Confidence and metadata
    extraction_confidence: float = Field(
        1.0, 
        description="Confidence score (0.0 to 1.0) for the extracted information"
    )
    extracted_from: str = Field(
        "", 
        description="Brief description of what triggered this extraction"
    )
    
    @validator('marital_status')
    def validate_marital_status(cls, v):
        """Validate marital status values."""
        if v is not None:
            valid_statuses = ['single', 'married', 'divorced', 'widowed', 'ledig', 'verheiratet']
            if v.lower() not in [s.lower() for s in valid_statuses]:
                # Don't fail, just log and normalize
                logging.warning(f"Unusual marital status: {v}")
        return v
    
    @validator('occupation')
    def validate_occupation(cls, v):
        """Normalize common German occupations."""
        if v is not None:
            # Common normalizations
            normalizations = {
                'freelancer': 'Freiberufler',
                'self-employed': 'Freiberufler',
                'employee': 'Angestellter',
                'civil servant': 'Beamter',
                'doctor': 'Freiberufler',  # Usually freelance in Germany
                'lawyer': 'Freiberufler',
                'consultant': 'Freiberufler'
            }
            return normalizations.get(v.lower(), v)
        return v


class NextAction(BaseModel):
    """Determines the next agent to call after updating the profile."""
    next_node: Literal["knowledge_agent", "end_conversation"] = Field(
        description="Next node: 'knowledge_agent' if question present, 'end_conversation' otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why this routing decision was made"
    )
    detected_question: bool = Field(
        description="Whether a question was detected in the user's message"
    )


# --- UTILITY FUNCTIONS ---
def setup_logging() -> logging.Logger:
    """Setup logging for the Profile Manager."""
    logger = logging.getLogger("profile_manager")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def format_conversation_history(messages: List[BaseMessage], max_length: int = 2000) -> str:
    """Format conversation history with length limiting."""
    if not messages:
        return "No conversation history."
    
    formatted = []
    total_length = 0
    
    # Process messages in reverse order (most recent first)
    for msg in reversed(messages):
        msg_text = f"{msg.type}: {msg.content}"
        if total_length + len(msg_text) > max_length:
            break
        formatted.insert(0, msg_text)  # Insert at beginning to maintain order
        total_length += len(msg_text)
    
    return "\n".join(formatted)


def validate_profile_updates(updates: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize profile updates."""
    validated_updates = {}
    
    for key, value in updates.items():
        if value is None:
            continue
            
        # Type-specific validation
        if key == "has_dependents" and not isinstance(value, bool):
            # Try to convert string to boolean
            if isinstance(value, str):
                value = value.lower() in ['true', '1', 'yes', 'ja']
            else:
                continue
        
        elif key in ["monthly_income", "annual_income"]:
            if isinstance(value, (int, float)) and value > 0:
                # --- START NEW LOGIC ---
                annual_income = 0
                if key == "monthly_income":
                    validated_updates["monthly_income"] = int(value)
                    annual_income = int(value) * 12
                    if "annual_income" not in updates:
                        validated_updates["annual_income"] = annual_income
                elif key == "annual_income":
                    validated_updates["annual_income"] = int(value)
                    annual_income = int(value)
                    if "monthly_income" not in updates:
                        validated_updates["monthly_income"] = int(value) // 12

                # Automatically set the income_range field
                if annual_income > 0:
                    if annual_income <= 30000:
                        validated_updates["income_range"] = "0-30k"
                    elif annual_income <= 60000:
                        validated_updates["income_range"] = "30k-60k"
                    elif annual_income <= 100000:
                        validated_updates["income_range"] = "60k-100k"
                    else:
                        validated_updates["income_range"] = "100k+"


        elif key == "known_expenses":
            if not isinstance(value, list):
                continue
            # Merge with existing expenses, avoiding duplicates
            current_expenses = current_profile.get("known_expenses", [])
            new_expenses = [exp for exp in value if exp not in current_expenses]
            if new_expenses:
                validated_updates[key] = current_expenses + new_expenses
            continue
        
        elif key in ["occupation", "marital_status"] and not isinstance(value, str):
            continue
        
        validated_updates[key] = value
    
    return validated_updates


# --- MAIN AGENT FUNCTIONS ---
async def extract_profile_information(
    messages: List[BaseMessage], 
    config: ProfileManagerConfig,
    current_profile: dict
) -> ProfileUpdates:
    """
    Extract profile information from conversation with enhanced context understanding.
    """
    logger = setup_logging()
    
    try:
        llm = ChatGroq(model=config.model_name, temperature=config.temperature)
        structured_llm = llm.with_structured_output(ProfileUpdates)
        
        # Enhanced extraction prompt with better German context
        extraction_prompt_text = """You are an expert at extracting user profile information for tax purposes.
Your goal is to find NEW or UPDATED information in the LATEST USER MESSAGE.

--- EXTRACTION HIERARCHY ---
1.  **EXPLICIT FACTS**: Prioritize information directly stated by the user.
2.  **LOGICAL INFERENCES**: Make common-sense connections based on user phrasing. This is crucial.
3.  **AVOID HALLUCINATION**: Do not invent information that isn't stated or strongly implied. It is better to have an empty field than an incorrect one.

--- LOGICAL INFERENCE RULES (VERY IMPORTANT) ---
-   If the user mentions "my wife", "my husband", or "my spouse", you MUST set `marital_status` to "married".
-   If the user mentions "my wife doesn't work" or "my spouse has no income", you MUST set `spouse_annual_income` to 0.
-   If the user mentions "my child", "my children", "my kids", "son", or "daughter", you MUST set `has_dependents` to true.

--- CRITICAL UPDATE RULES ---
1. **UPDATE EXISTING FIELDS**: If user provides NEW information that contradicts existing profile:
   - "I earn 3000€ monthly" should UPDATE monthly_income even if one exists
   - "I work in graphic design" should UPDATE or ADD to occupation details
   - "I'm actually single" should UPDATE marital_status from married to single

2. **INCREMENTAL INFORMATION**: Add details to existing fields:
   - If occupation is "Freiberufler", and user says "graphic design" → "Freiberufler - Graphic Design"

--- EXTRACTION RULES ---
1. **Data Types**: 
   - `has_dependents`: MUST be boolean (true/false), not string
   - `marital_status`: MUST be string ("married", "single", etc.)
   - `known_expenses`: MUST be list of strings (expense categories/items)
   - `is_kleinunternehmer`: MUST be boolean if mentioned
   - `monthly_income`: MUST be integer (euros per month) if mentioned
   - `annual_income`: MUST be integer (euros per year) if mentioned

2. **Income Detection**: 
   - "I earn 3000€ per month" → monthly_income: 3000, annual_income: 36000
   - "My annual income is 45000€" → annual_income: 45000, monthly_income: 3750
   - "I make about 4000€ monthly" → monthly_income: 4000, annual_income: 48000

3. **German Tax Context**:
   - "Freiberufler" = freelancer/self-employed professional
   - "Angestellter" = employee
   - "Beamter" = civil servant
   - "Kleinunternehmer" = small business (VAT exemption)

4. **Dependency Detection**:
   - Wife/husband/spouse → `marital_status`: "married"
   - Children/kids/son/daughter → `has_dependents`: true
   - Family → could indicate dependents (use judgment)

5. **Expense Detection**:
   - Look for specific costs, purchases, business expenses
   - Extract the category/type, not just amounts
   - Examples: "Home office equipment", "Business travel", "Professional training"

6. **Confidence Scoring**:
   - 1.0 = Explicitly stated ("I am a freelancer")
   - 0.8 = Strongly implied ("As a doctor..." - likely Freiberufler)
   - 0.6 = Moderately implied (context clues)
   - 0.4 = Weakly implied (assumptions)
   - Set to 0.0 if uncertain

--- EXAMPLES ---
User: "I'm a freelancer and bought a new laptop for 1200€"
→ occupation: "Freiberufler", known_expenses: ["Laptop purchase"], confidence: 1.0

User: "My wife and I have two children"
→ marital_status: "married", has_dependents: true, confidence: 1.0

User: "As a doctor, I work from my home office"
→ occupation: "Freiberufler", confidence: 0.8 (doctors are usually freelance)


User: "I'm married and my wife doesn't work. We have one child."
→ marital_status: "married", spouse_annual_income: 0, has_dependents: true, confidence: 1.0

User: "As a doctor, I work from my home office"
→ occupation: "Freiberufler", confidence: 0.8 (doctors are usually freelance)
"""
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", extraction_prompt_text),
            ("human", "Analyze this conversation and extract profile information:\n\n{conversation}"),
        ])
        
        extraction_chain = extraction_prompt | structured_llm
        
        # Format conversation for analysis
        conversation_text = format_conversation_history(messages)
        
        # Execute extraction
        extracted_data = await extraction_chain.ainvoke({
            "conversation": conversation_text,
            "current_profile": str(current_profile)
        })
        
        if config.enable_debug_logging:
            logger.info(f"Extracted profile data: {extracted_data.dict(exclude_unset=True)}")
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}")
        # Return empty extraction on failure
        return ProfileUpdates(
            extraction_confidence=0.0,
            extracted_from=f"extraction_error: {str(e)}"
        )


async def determine_next_action(
    last_message: str, 
    config: ProfileManagerConfig
) -> NextAction:
    """
    Determine the next routing action based on the user's message.
    """
    logger = setup_logging()
    
    try:
        llm = ChatGroq(model=config.model_name, temperature=config.temperature)
        structured_llm = llm.with_structured_output(NextAction)
        
        routing_prompt_text = """You are an expert at analyzing user messages in tax context to determine routing.

--- ROUTING RULES ---
1. **Priority: Detect Questions (Even After Profile Updates)**
   - Look for: question words (how, what, why, when, where, which, should, can, could)
   - Look for: question marks (?)
   - Look for: implicit questions ("I want to know about...", "Tell me about...")
   - Look for: help requests ("Can you help with...", "I need info on...")
   - Look for: decision requests ("Should I...", "What's better...", "Which option...")

2. **IMPORTANT: Complex Messages**
   - If message contains BOTH profile info AND a question → knowledge_agent
   - Example: "I'm a freelancer earning 4500€. Should I register for VAT?" → knowledge_agent
   - Example: "I'm married with two children. What deductions apply?" → knowledge_agent

3. **Question Types in Tax Context**:
   - Direct questions: "How does home office deduction work?"
   - Implicit questions: "I want to know about Kleinunternehmer rules"
   - Help requests: "Can you explain ELSTER?"
   - Information seeking: "Tell me about tax deadlines"

4. **Route to 'knowledge_agent' if**:
   - Any question detected (direct or implicit)
   - Information seeking behavior
   - Requests for explanations or help

5. **Route to 'end_conversation' if**:
   - Pure statements with no questions
   - Simple confirmations ("Yes", "OK", "Thanks")
   - Profile updates only ("I am married", "My job is...")

--- EXAMPLES ---
"I'm a freelancer earning 4500€. Should I register for VAT?" → knowledge_agent (question after profile info)
"As a Freiberufler, how does home office work?" → knowledge_agent (question detected)
"I want to know about tax deadlines" → knowledge_agent (information seeking)
"I am a doctor with two children" → end_conversation (pure statement)
"Thanks, that helps!" → end_conversation (acknowledgment)
"""
        
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", routing_prompt_text),
            ("human", "Analyze this message and determine routing:\n\nMessage: {message}"),
        ])
        
        routing_chain = routing_prompt | structured_llm
        
        result = await routing_chain.ainvoke({"message": last_message})
        
        if config.enable_debug_logging:
            logger.info(f"Routing decision: {result.next_node} - {result.reasoning}")
        
        return result
        
    except Exception as e:
        logger.error(f"Routing decision failed: {e}")
        # Default to knowledge_agent on failure (safer)
        return NextAction(
            next_node="knowledge_agent",
            reasoning=f"Error in routing, defaulting to knowledge_agent: {str(e)}",
            detected_question=True
        )


def generate_confirmation_message(updates: Dict[str, Any]) -> Optional[str]:
    """Generate a user-friendly confirmation message for profile updates."""
    if not updates:
        return None
    
    confirmations = []

    if "monthly_income" in updates:
        confirmations.append(f"monthly income: {updates['monthly_income']}€")
    if "annual_income" in updates and "monthly_income" not in updates:
        confirmations.append(f"annual income: {updates['annual_income']}€")
    
    # Create human-readable confirmations
    field_descriptions = {
        "occupation": "occupation",
        "marital_status": "marital status", 
        "has_dependents": "dependent information",
        "is_kleinunternehmer": "Kleinunternehmer status",
        "known_expenses": "expense records"
    }
    
    for field, value in updates.items():
        if field in field_descriptions:
            desc = field_descriptions[field]
            
            if field == "known_expenses" and isinstance(value, list):
                if len(value) == 1:
                    confirmations.append(f"recorded your {value[0]}")
                else:
                    confirmations.append(f"recorded {len(value)} expense items")
            elif field == "has_dependents":
                status = "with dependents" if value else "without dependents"
                confirmations.append(f"noted you are {status}")
            else:
                confirmations.append(f"updated your {desc}")
    
    if confirmations:
        if len(confirmations) == 1:
            return f"Got it! I've {confirmations[0]}."
        else:
            return f"Got it! I've {', '.join(confirmations[:-1])} and {confirmations[-1]}."
    
    return "Thanks! I've updated your profile."


# --- MAIN AGENT ENTRY POINT ---
async def run_profile_manager(state: GraphState) -> Dict[str, Any]:
    """
    Main entry point for the Profile Manager Agent. (Updated Logic)
    
    This function:
    1. Extracts profile information from the conversation.
    2. Updates the user profile with validated data.
    3. Generates a confirmation message for any new data.
    4. Ends the turn with the confirmation if no further question is asked.
    """
    logger = setup_logging()
    config = ProfileManagerConfig()
    
    logger.info("--- Running Profile Manager Agent ---")
    
    try:
        if not state.get("messages"):
            logger.warning("No messages in state")
            return {"next_node": "end_conversation"}
        
        messages = state["messages"]
        current_profile = state.get("user_profile", {}).copy()
        latest_message = messages[-1].content if messages else ""
        
        # Step 1: Extract profile information
        # (Assuming you have also applied the recommended prompt fix here)
        logger.info("Extracting profile information...")
        extracted_data = await extract_profile_information(messages, config, current_profile)
        
        # Step 2: Process, validate, and identify new updates
        raw_updates = extracted_data.dict(exclude_unset=True)
        profile_updates = {k: v for k, v in raw_updates.items() if k not in ['extraction_confidence', 'extracted_from']}
        validated_updates = validate_profile_updates(profile_updates, current_profile)
        
        truly_new_updates = {}
        for key, value in validated_updates.items():
            if key == "known_expenses":
                current_expenses = set(current_profile.get("known_expenses", []))
                new_expenses = [exp for exp in value if exp not in current_expenses]
                if new_expenses:
                    truly_new_updates[key] = new_expenses
            elif current_profile.get(key) != value:
                truly_new_updates[key] = value

        updated_profile = current_profile.copy()
        updated_profile.update(validated_updates)
        
        if truly_new_updates:
            logger.info(f"New profile updates detected: {truly_new_updates}")
        else:
            logger.info("No new profile updates detected")
        
        # Step 3: Determine if the user also asked a question
        logger.info("Determining next action...")
        next_action = await determine_next_action(latest_message, config)

        # Step 4: Generate confirmation message for new updates
        confirmation = None
        if truly_new_updates:
            confirmation = generate_confirmation_message(truly_new_updates)

        # Step 5: **REVISED LOGIC** - Decide what to return based on the context
        
        # SCENARIO A: A profile update happened, and there was NO question.
        # End the conversation turn immediately with the confirmation message.
        if confirmation and next_action.next_node == "end_conversation":
            logger.info(f"Profile updated. Ending turn with confirmation: '{confirmation}'")
            return {
                "user_profile": updated_profile,
                "next_node": "end_conversation",  # This stops the agent chain
                "direct_response": confirmation
            }
        
        # SCENARIO B: A profile update happened, AND a question was also asked.
        # Update the profile and route to the knowledge agent to answer the question.
        if next_action.next_node == "knowledge_agent":
            logger.info("Profile updated, but a question was also detected. Routing to knowledge_agent.")
            return {
                "user_profile": updated_profile,
                "next_node": "knowledge_agent",
                # We can optionally pass the confirmation for the UI to display first
                "direct_response": confirmation 
            }
            
        # SCENARIO C: No updates were made and no question was asked.
        # This will now only happen for simple inputs like "thanks" or "ok".
        logger.info(f"No profile updates. Following router decision: {next_action.next_node}")
        return {
            "user_profile": updated_profile,
            "next_node": next_action.next_node
        }
        
    except Exception as e:
        logger.error(f"Profile Manager failed: {e}")
        return {
            "user_profile": state.get("user_profile", {}),
            "next_node": "knowledge_agent",
            "direct_response": "I encountered an issue updating your profile, but I can still help with your questions."
        }