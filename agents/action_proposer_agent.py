# agents/action_proposer_agent.py
"""
Action Proposer Agent - Proactive assistant for suggesting next steps to users.

This agent analyzes the user's profile after questions are answered and suggests
logical next steps to improve their tax situation or complete missing information.
"""

import logging
from typing import Literal, Optional, Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from graph_state import GraphState

# Configure logging
logger = logging.getLogger(__name__)


class ProposedAction(BaseModel):
    """
    Structured model for proposed user actions.
    
    This ensures the LLM provides predictable, actionable suggestions
    that the UI can render as interactive elements.
    """
    action_type: Literal[
        "add_expense", 
        "review_deductions",
        "add_income", 
        "update_income", 
        "add_dependent", 
        "check_tax_class",
        "schedule_appointment",
        "none"
    ] = Field(
        description="The type of action to propose to the user"
    )
    
    rationale: Optional[str] = Field(
        description="User-friendly explanation of why this action is suggested"
    )
    
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Priority level of the suggested action"
    )
    
    estimated_benefit: Optional[str] = Field(
        description="Potential tax benefit or impact (e.g., '€500-1000 potential savings')"
    )
    
    action_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional data needed for the action (e.g., expense categories)"
    )


class ActionProposerAgent:
    """
    Agent responsible for analyzing user profiles and suggesting proactive actions.
    
    This agent helps guide users through the tax process by identifying gaps
    in their information and opportunities for optimization.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.1):
        """
        Initialize the Action Proposer Agent.
        
        Args:
            model_name: Groq model to use for action suggestions
            temperature: Slightly higher temperature for more creative suggestions
        """
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._structured_llm = None
    
    @property
    def llm(self):
        """Lazy initialization of the LLM."""
        if self._llm is None:
            try:
                self._llm = ChatGroq(model=self.model_name, temperature=self.temperature)
                self._structured_llm = self._llm.with_structured_output(ProposedAction)
            except Exception as e:
                logger.error(f"Failed to initialize Action Proposer LLM: {e}")
                raise
        return self._structured_llm
    
    def _analyze_profile_gaps(self, user_profile: Dict[str, Any]) -> List[str]:
        """
        Analyze the user profile for missing information or opportunities.
    
        Args:
            user_profile: Current user profile data
        
        Returns:
            List of identified gaps or opportunities
        """
        gaps = []
    
        # Check for missing basic information
        if not user_profile.get("occupation"):
            gaps.append("missing_occupation")
    
        if not user_profile.get("income_range"):
            gaps.append("missing_income")
    
        if user_profile.get("marital_status") == "married" and not user_profile.get("tax_class"):
            gaps.append("missing_tax_class")
    
        # Income gap detection (avoid duplicate gaps)
        if not user_profile.get("monthly_income") and not user_profile.get("annual_income"):
            if "missing_income" not in gaps:  # Avoid duplicate
                gaps.append("missing_income")
    
        # Check for missing expenses based on occupation - WITH NULL SAFETY
        occupation = user_profile.get("occupation")
        if occupation and isinstance(occupation, str):  # NULL SAFETY CHECK
            occupation_lower = occupation.lower()
            known_expenses = user_profile.get("known_expenses", [])
        
            if "freelancer" in occupation_lower or "freiberufler" in occupation_lower or "selbstständig" in occupation_lower:
                if not known_expenses:
                    gaps.append("missing_business_expenses")
        
            if "employee" in occupation_lower or "angestellt" in occupation_lower:
                common_deductions = ["commuting", "home_office", "professional_development"]
                
                # Safe expense checking with null protection
                has_common_deduction = False
                if known_expenses:
                    try:
                        for expense in known_expenses:
                            if expense and isinstance(expense, str):  # NULL SAFETY
                                expense_lower = expense.lower()
                                if any(deduction in expense_lower for deduction in common_deductions):
                                    has_common_deduction = True
                                    break
                    except Exception as e:
                        logger.warning(f"Error checking expenses: {e}")
                        # Continue without this check
                
                if not has_common_deduction:
                    gaps.append("missing_employee_deductions")
        else:
            # If occupation is None or not a string, we definitely have a gap
            if "missing_occupation" not in gaps:
                gaps.append("missing_occupation")
    
        # Check for family-related opportunities
        if user_profile.get("has_children") and not user_profile.get("child_benefits_claimed"):
            gaps.append("missing_child_benefits")
    
        return gaps
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """
        Build the action proposal prompt with comprehensive rules and examples.
        
        Returns:
            ChatPromptTemplate for action suggestion
        """
        system_prompt = """You are a proactive tax assistant that suggests helpful next steps.
Analyze the user's profile to identify the most valuable action they could take next.

ANALYSIS FRAMEWORK:

1. **Profile Completeness**: Identify missing critical information
2. **Optimization Opportunities**: Find potential tax savings
3. **Process Guidance**: Suggest logical next steps in tax preparation
4. **Compliance**: Ensure all required information is captured

ACTION PRIORITY RULES:

**HIGH PRIORITY** (immediate attention needed):
- Missing income information for tax calculation
- Freelancers/self-employed without any recorded expenses
- Married couples without tax class optimization
- Missing child benefit claims for families

**MEDIUM PRIORITY** (good opportunities):
- Employees missing common deductions (commuting, home office)
- Incomplete expense documentation
- Missing professional development costs
- Unoptimized deduction strategies

**LOW PRIORITY** (nice to have):
- Additional minor deductions
- Future planning suggestions
- Process optimization tips

SPECIFIC ACTION TYPES:

- **add_expense**: When business expenses or deductions are missing
- **update_income**: When income information is incomplete/missing
- **add_dependent**: When family members aren't recorded for benefits
- **check_tax_class**: When married couples need tax class optimization
- **review_deductions**: When existing deductions could be optimized
- **schedule_appointment**: For complex situations needing professional help
- **none**: When profile is complete and well-optimized

EXAMPLES:

Profile: {{"occupation": "freelancer", "known_expenses": [], "income_range": "30000-50000"}}
→ Action: add_expense, Priority: high, Rationale: "As a freelancer, recording business expenses could save you significant taxes"

Profile: {{"occupation": "employee", "known_expenses": ["Home Office"], "marital_status": "married", "tax_class": null}}
→ Action: check_tax_class, Priority: high, Rationale: "Married couples can often save taxes by optimizing their tax class combination"

Profile: {{"occupation": "employee", "known_expenses": ["Commuting", "Home Office", "Professional Development"], "income_range": "40000-60000"}}
→ Action: none, Priority: low, Rationale: "Your tax profile looks well-optimized"

Profile: {{"occupation": "freelancer", "known_expenses": [], "monthly_income": null}}
→ Action: add_income, Priority: high, Rationale: "As a freelancer, income information is crucial for tax planning"

Profile: {{"occupation": "freelancer", "known_expenses": [], "monthly_income": 3500}}
→ Action: add_expense, Priority: high, Rationale: "With your income level, business expenses could provide significant tax savings"

Always consider the tax context and provide actionable, specific suggestions."""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Please analyze this user profile and suggest the most valuable next action:

User Profile: {user_profile}

Identified gaps: {profile_gaps}

Consider the user's current situation and suggest an action that would provide the most benefit.""")
        ])
    
    async def propose_action(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyze user profile and propose the most valuable next action.
    
        Args:
            state: Current graph state containing user profile
        
        Returns:
            Dictionary with proposed action or empty dict if no action needed
        """
        logger.info("--- Running Action Proposer ---")
    
        try:
            # Validate state and user profile
            user_profile = state.get("user_profile", {})
            messages = state.get("messages", [])
            
            if not user_profile:
                logger.warning("No user profile found in state")
                return {}
            
            # Safe message processing to avoid null errors
            last_user_message = ""
            if messages:
                try:
                    for message in reversed(messages):
                        # Safe attribute access with null checks
                        message_type = getattr(message, 'type', None)
                        message_content = getattr(message, 'content', None)
                        
                        if (message_type == 'human' and 
                            message_content and 
                            isinstance(message_content, str) and 
                            message_content.strip()):
                            
                            last_user_message = message_content.strip().lower()
                            break
                except Exception as msg_error:
                    logger.warning(f"Error processing messages: {msg_error}")
                    # Continue without last message context
            
            logger.info(f"Processing last message: {last_user_message[:100]}...")
        
            # Analyze profile for gaps and opportunities
            profile_gaps = self._analyze_profile_gaps(user_profile)
            logger.info(f"Identified profile gaps: {profile_gaps}")
            
            # Handle action cooldown to avoid repetitive suggestions
            last_action_type = state.get("last_proposed_action_type")
            if last_action_type:
                logger.info(f"Last action type was: {last_action_type}")
                
                # Remove gaps that were already suggested
                original_gaps = len(profile_gaps)
                if last_action_type == "update_income":
                    profile_gaps = [g for g in profile_gaps if g != "missing_income"]
                elif last_action_type == "check_tax_class":
                    profile_gaps = [g for g in profile_gaps if g != "missing_tax_class"]
                elif last_action_type == "add_expense":
                    profile_gaps = [g for g in profile_gaps if g != "missing_business_expenses"]
                elif last_action_type == "add_income":
                    profile_gaps = [g for g in profile_gaps if g != "missing_income"]
                
                filtered_gaps = len(profile_gaps)
                if original_gaps != filtered_gaps:
                    logger.info(f"Filtered out repeated suggestions: {original_gaps} → {filtered_gaps} gaps")
        
            # If no significant gaps remain, don't propose unnecessary actions
            if not profile_gaps:
                logger.info("No significant gaps found after filtering, no action needed")
                return {}
        
            # Build and execute the proposal chain
            try:
                prompt = self._build_prompt()
                chain = prompt | self.llm
            
                proposed_action = await chain.ainvoke({
                    "user_profile": user_profile,
                    "profile_gaps": profile_gaps
                })
                
                # Validate the proposed action structure
                if not proposed_action:
                    logger.warning("LLM returned empty action")
                    return {}
                    
                if not hasattr(proposed_action, 'action_type'):
                    logger.warning("LLM response missing action_type")
                    return {}
            
            except Exception as llm_error:
                logger.error(f"LLM execution failed: {llm_error}")
                return {}
        
            # Validate the proposed action content
            if proposed_action.action_type == "none":
                logger.info("LLM determined no action is needed")
                return {}
        
            logger.info(f"Proposed action: {proposed_action.action_type} (priority: {proposed_action.priority})")
            if proposed_action.rationale:
                logger.info(f"Rationale: {proposed_action.rationale}")
        
            # Build structured action data with safe attribute access
            action_data = {
                "action_type": getattr(proposed_action, 'action_type', 'add_expense'),
                "rationale": getattr(proposed_action, 'rationale', 'No rationale provided'),
                "priority": getattr(proposed_action, 'priority', 'medium'),
                "estimated_benefit": getattr(proposed_action, 'estimated_benefit', None),
                "action_data": getattr(proposed_action, 'action_data', {}),
                "timestamp": state.get("timestamp", ""),
            }
            
            # Final validation of action_data
            if not action_data.get("action_type") or not action_data.get("rationale"):
                logger.warning("Invalid action data structure, skipping action proposal")
                return {}
        
            return {"proposed_action": action_data}
        
        except Exception as e:
            logger.error(f"Error in action proposer: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return empty dict to fail gracefully without breaking conversation flow
            return {}
    
    def _get_expense_categories_for_occupation(self, occupation: str) -> List[str]:
        """
        Get relevant expense categories based on user's occupation.
        
        Args:
            occupation: User's occupation
            
        Returns:
            List of relevant expense categories
        """
        occupation = occupation.lower()
        
        if "freelancer" in occupation or "selbstständig" in occupation:
            return [
                "Home Office",
                "Professional Equipment",
                "Software & Subscriptions", 
                "Internet & Phone",
                "Professional Development",
                "Business Travel",
                "Client Entertainment"
            ]
        elif "employee" in occupation or "angestellt" in occupation:
            return [
                "Commuting Costs",
                "Home Office Expenses",
                "Professional Development",
                "Work Equipment",
                "Professional Literature"
            ]
        else:
            return [
                "Professional Expenses",
                "Education & Training",
                "Equipment & Tools"
            ]


# Global instance for backward compatibility
_action_proposer_instance = ActionProposerAgent()


async def run_action_proposer(state: GraphState) -> Dict[str, Any]:
    """
    Main entry point for the action proposer agent (backward compatible).
    
    Args:
        state: GraphState containing user profile
        
    Returns:
        Dictionary with proposed action or empty dict
    """
    return await _action_proposer_instance.propose_action(state)


def get_action_proposer_instance() -> ActionProposerAgent:
    """Get the global action proposer instance for direct access."""
    return _action_proposer_instance