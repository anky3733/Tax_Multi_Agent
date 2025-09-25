# tools/profile_tools.py
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List

class ProfileUpdates(BaseModel):
    """A model to hold extracted user profile information."""
    occupation: Optional[str] = Field(
        description="The user's occupation, e.g., 'freelancer', 'doctor', 'engineer'."
    )
    marital_status: Optional[str] = Field(
        description="The user's marital status, e.g., 'single', 'married'."
    )
    has_dependents: Optional[bool] = Field(
        description="Whether the user has dependents."
    )
    known_expenses: Optional[List[str]] = Field(
        description="A list of specific expenses the user has mentioned."
    )