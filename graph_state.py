# graph_state.py
"""
Graph State Definition - Core data structures for the tax assistance system.

This module defines the TypedDict structures that represent the complete state
of the multi-agent conversation system, including user profile and conversation context.
"""

from typing import TypedDict, List, Annotated, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage
from datetime import datetime
import operator


class TaxDocuments(TypedDict, total=False):
    """Tax-related documents and their processing status."""
    w2_forms: List[str]  # List of document IDs or file paths
    receipts: List[str]
    contracts: List[str]
    bank_statements: List[str]
    other_documents: List[str]


class TaxYearData(TypedDict, total=False):
    """Tax information for a specific year."""
    year: int
    income: Optional[float]
    expenses: Dict[str, float]  # Category -> Amount
    deductions: Dict[str, float]  # Deduction type -> Amount
    documents: TaxDocuments
    filing_status: Optional[str]  # "draft", "submitted", "processed"
    estimated_refund: Optional[float]


class UserProfile(TypedDict, total=False):
    """
    Comprehensive user profile representing long-term memory for German tax system.
    
    This profile persists across sessions and builds up over time as the system
    learns more about the user's tax situation.
    """
    # Basic Demographics
    user_id: Optional[str]
    first_name: Optional[str]
    occupation: Optional[str]  # e.g., "Freiberufler", "Angestellter", "Beamter"
    employment_type: Optional[str]  # "employed", "self_employed", "freelancer", "retired"
    
    # Personal Status
    marital_status: Optional[str]  # "single", "married", "divorced", "widowed"
    tax_class: Optional[int]  # German tax classes 1-6
    has_dependents: bool
    number_of_children: int
    children_ages: List[int]
    
    # Income Information
    annual_income: Optional[float]
    income_range: Optional[str]  # "0-30k", "30k-60k", "60k-100k", "100k+"
    income_sources: List[str]  # ["salary", "freelance", "rental", "investment"]
    
    # Business Information (for self-employed/freelancers)
    is_kleinunternehmer: Optional[bool]  # Small business VAT exemption
    business_type: Optional[str]
    vat_registered: Optional[bool]
    business_start_date: Optional[str]
    
    # Expenses and Deductions
    known_expenses: List[str]  # Free-text expense descriptions
    expense_categories: Dict[str, List[Dict[str, Any]]]  # Category -> [{amount, description, date}]
    regular_deductions: Dict[str, float]  # Monthly recurring deductions
    
    # Tax History
    previous_years: Dict[str, TaxYearData]  # Year -> Tax data
    last_filing_year: Optional[str]
    typical_refund_amount: Optional[float]
    
    # Preferences and Settings
    preferred_language: str  # "de", "en"
    notification_preferences: Dict[str, bool]
    risk_tolerance: str  # "conservative", "moderate", "aggressive"
    
    # System Metadata
    profile_created: Optional[str]  # ISO datetime string
    last_updated: Optional[str]
    profile_completeness: float  # 0.0 to 1.0
    data_version: str  # For schema migration


class ConversationContext(TypedDict, total=False):
    """Context information for the current conversation session."""
    session_id: Optional[str]
    conversation_start: Optional[str]  # ISO datetime
    last_activity: Optional[str]
    topic_focus: Optional[str]  # Current conversation topic
    user_intent: Optional[str]  # Inferred user intent
    conversation_stage: Optional[str]  # "greeting", "info_gathering", "answering", "action"
    questions_asked: List[str]  # Track questions asked to avoid repetition
    topics_covered: List[str]  # Track covered topics


class AgentMetadata(TypedDict, total=False):
    """Metadata about agent processing and routing."""
    last_agent: Optional[str]  # Which agent handled the last message
    agent_confidence: Optional[float]  # Confidence of last routing decision
    processing_time: Optional[float]  # Time taken by last agent
    routing_history: List[Dict[str, Any]]  # History of routing decisions
    error_count: int  # Number of errors in current session


class ResponseMetadata(TypedDict, total=False):
    """Metadata about system responses."""
    response_type: Optional[str]  # "answer", "question", "action_proposal"
    confidence_score: Optional[float]
    sources_used: List[str]  # Knowledge sources referenced
    personalization_applied: List[str]  # What personalizations were used
    reasoning_steps: List[str]  # Steps taken to arrive at response


class ProposedAction(TypedDict, total=False):
    """Structure for proposed user actions."""
    action_type: str  # Type of action proposed
    rationale: str  # Why this action is suggested
    priority: str  # "high", "medium", "low"
    estimated_benefit: Optional[str]  # Potential benefit description
    action_data: Dict[str, Any]  # Additional data for the action
    expires_at: Optional[str]  # When this suggestion becomes stale
    timestamp: str  # When this action was proposed


class GraphState(TypedDict):
    """
    Complete state representation for the multi-agent tax assistance system.
    
    This state is passed between all agents and contains all information
    needed to provide personalized, contextual responses.
    """
    # Core conversation data
    messages: Annotated[List[BaseMessage], operator.add]
    
    # User information and memory
    user_profile: UserProfile
    conversation_context: ConversationContext
    
    # Routing and control flow
    next_node: Optional[str]  # Which agent to call next
    previous_node: Optional[str]  # Which agent was called last
    
    # Response data
    direct_response: Optional[str]  # Direct answer from knowledge agent
    proposed_action: Optional[ProposedAction]  # Action suggested by action proposer
    last_proposed_action_type: Optional[str]  # Type of last proposed action
    
    # System metadata
    agent_metadata: AgentMetadata
    response_metadata: ResponseMetadata
    
    # Error handling
    error_message: Optional[str]  # Last error message
    retry_count: int  # Number of retries for current operation
    
    # Session management
    session_active: bool  # Whether session is still active
    requires_user_input: bool  # Whether system is waiting for user input


class MinimalGraphState(TypedDict):
    """
    Minimal state for lightweight operations or testing.
    Contains only the essential fields needed for basic functionality.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: UserProfile
    next_node: Optional[str]


def create_empty_user_profile() -> UserProfile:
    """
    Create an empty user profile with default values.
    
    Returns:
        UserProfile with sensible defaults
    """
    return UserProfile(
        has_dependents=False,
        number_of_children=0,
        children_ages=[],
        known_expenses=[],
        expense_categories={},
        regular_deductions={},
        previous_years={},
        preferred_language="de",
        notification_preferences={},
        risk_tolerance="moderate",
        profile_completeness=0.0,
        data_version="1.0"
    )


def create_initial_graph_state(user_id: Optional[str] = None) -> GraphState:
    """
    Create initial graph state for a new conversation.
    
    Args:
        user_id: Optional user ID for profile association
        
    Returns:
        Initialized GraphState
    """
    now = datetime.now().isoformat()
    
    user_profile = create_empty_user_profile()
    if user_id:
        user_profile["user_id"] = user_id
        user_profile["profile_created"] = now
    
    return GraphState(
        messages=[],
        user_profile=user_profile,
        conversation_context=ConversationContext(
            conversation_start=now,
            last_activity=now,
            conversation_stage="greeting",
            questions_asked=[],
            topics_covered=[]
        ),
        next_node=None,
        previous_node=None,
        direct_response=None,
        proposed_action=None,
        agent_metadata=AgentMetadata(
            routing_history=[],
            error_count=0
        ),
        response_metadata=ResponseMetadata(
            sources_used=[],
            personalization_applied=[],
            reasoning_steps=[]
        ),
        error_message=None,
        retry_count=0,
        session_active=True,
        requires_user_input=False
    )


def calculate_profile_completeness(profile: UserProfile) -> float:
    """
    Calculate how complete a user profile is (0.0 to 1.0).
    
    Args:
        profile: User profile to evaluate
        
    Returns:
        Completeness score between 0.0 and 1.0
    """
    required_fields = [
        "occupation", "marital_status", "annual_income", 
        "tax_class", "employment_type"
    ]
    
    optional_fields = [
        "business_type", "known_expenses", "expense_categories"
    ]
    
    required_score = sum(1 for field in required_fields if profile.get(field)) / len(required_fields)
    optional_score = sum(1 for field in optional_fields if profile.get(field)) / len(optional_fields)
    
    # Weight required fields more heavily (70% vs 30%)
    return (required_score * 0.7) + (optional_score * 0.3)


def update_profile_metadata(profile: UserProfile) -> UserProfile:
    """
    Update profile metadata like completeness score and last_updated timestamp.
    
    Args:
        profile: Profile to update
        
    Returns:
        Updated profile
    """
    now = datetime.now().isoformat()
    profile["last_updated"] = now
    profile["profile_completeness"] = calculate_profile_completeness(profile)
    return profile