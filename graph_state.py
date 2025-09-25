from typing import TypedDict, List, Annotated
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

    Attributes:
        messages: The conversation history.
        user_profile: The user's persistent profile.
        next_node: The next agent to call.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: UserProfile
    next_node: str