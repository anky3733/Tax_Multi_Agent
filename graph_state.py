from typing import TypedDict, List, Annotated, Optional, Dict
from langchain_core.messages import BaseMessage
import operator

class UserProfile(TypedDict):
    """Represents the long-term memory of the user."""
    occupation: str = None
    marital_status: str = None
    has_dependents: bool = False
    known_expenses: List[str] = []

class GraphState(TypedDict):
    """
    Represents the entire state of our graph.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: UserProfile
    next_node: str
    # --- NEW FIELD ---
    # final_response: Optional[str] = None

    direct_response: Optional[str] = None
    proposed_action: Optional[Dict] = None