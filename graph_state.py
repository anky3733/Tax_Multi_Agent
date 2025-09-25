from typing import TypedDict, List, Annotated, Optional, Dict
from langchain_core.messages import BaseMessage
import operator

class UserProfile(TypedDict):
    """Represents the long-term memory of the user for the German tax system."""
    occupation: str = None         # e.g., "Freiberufler", "Angestellter"
    marital_status: str = None     # e.g., "ledig", "verheiratet"
    has_dependents: bool = False
    is_kleinunternehmer: bool = None # Is the user a "small business" for VAT?
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