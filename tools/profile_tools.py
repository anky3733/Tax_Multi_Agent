# tools/profile_tools.py
"""
Profile Tools - Data extraction and management for user profiles.

This module provides structured models and utilities for extracting, validating,
and managing user profile information from conversations in the German tax context.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)


class OccupationType(str, Enum):
    """Standardized occupation types for German tax system."""
    EMPLOYEE = "employee"  # Angestellter
    FREELANCER = "freelancer"  # Freiberufler
    SELF_EMPLOYED = "self_employed"  # Selbstständiger
    CIVIL_SERVANT = "civil_servant"  # Beamter
    PENSIONER = "pensioner"  # Rentner
    STUDENT = "student"  # Student
    UNEMPLOYED = "unemployed"  # Arbeitslos
    OTHER = "other"


class MaritalStatus(str, Enum):
    """German marital status options."""
    SINGLE = "single"  # Ledig
    MARRIED = "married"  # Verheiratet
    DIVORCED = "divorced"  # Geschieden
    WIDOWED = "widowed"  # Verwitwet
    SEPARATED = "separated"  # Getrennt lebend


class TaxClass(int, Enum):
    """German tax classes (Steuerklassen)."""
    CLASS_1 = 1  # Single, divorced, widowed
    CLASS_2 = 2  # Single with children
    CLASS_3 = 3  # Married, higher earner
    CLASS_4 = 4  # Married, both work
    CLASS_5 = 5  # Married, lower earner
    CLASS_6 = 6  # Second job


class ExpenseCategory(str, Enum):
    """Common German tax deductible expense categories."""
    HOME_OFFICE = "home_office"  # Homeoffice-Pauschale
    COMMUTING = "commuting"  # Entfernungspauschale
    PROFESSIONAL_DEVELOPMENT = "professional_development"  # Fortbildung
    WORK_EQUIPMENT = "work_equipment"  # Arbeitsmittel
    BUSINESS_MEALS = "business_meals"  # Bewirtungskosten
    TRAVEL_EXPENSES = "travel_expenses"  # Reisekosten
    INTERNET_PHONE = "internet_phone"  # Internet/Telefon
    PROFESSIONAL_LITERATURE = "professional_literature"  # Fachliteratur
    SOFTWARE_SUBSCRIPTIONS = "software_subscriptions"  # Software-Abos
    INSURANCE = "insurance"  # Versicherungen
    OTHER = "other"


class ExpenseItem(BaseModel):
    """Individual expense item with validation."""
    description: str = Field(description="Description of the expense")
    category: Optional[ExpenseCategory] = Field(description="Expense category")
    amount: Optional[float] = Field(description="Amount in EUR", ge=0)
    frequency: Optional[str] = Field(
        description="How often this expense occurs: 'once', 'monthly', 'yearly'"
    )
    date_incurred: Optional[str] = Field(description="When this expense was incurred (YYYY-MM-DD)")
    tax_year: Optional[int] = Field(description="Tax year this expense applies to", ge=2020, le=2030)
    
    @validator('amount')
    def validate_amount(cls, v):
        if v is not None and v < 0:
            raise ValueError('Amount must be non-negative')
        return v
    
    @validator('frequency')
    def validate_frequency(cls, v):
        if v is not None and v not in ['once', 'monthly', 'yearly', 'weekly']:
            raise ValueError('Frequency must be one of: once, monthly, yearly, weekly')
        return v


class IncomeSource(BaseModel):
    """Income source with details."""
    source_type: str = Field(description="Type of income: salary, freelance, rental, investment")
    amount: Optional[float] = Field(description="Annual amount in EUR", ge=0)
    employer_name: Optional[str] = Field(description="Name of employer or client")
    is_primary: bool = Field(default=False, description="Whether this is the primary income source")


class DependentInfo(BaseModel):
    """Information about dependents for tax purposes."""
    relationship: str = Field(description="Relationship: child, spouse, parent, etc.")
    age: Optional[int] = Field(description="Age of dependent", ge=0, le=120)
    eligible_for_child_benefit: Optional[bool] = Field(
        description="Whether eligible for Kindergeld"
    )
    in_education: Optional[bool] = Field(description="Whether currently in education/training")


class ProfileUpdates(BaseModel):
    """
    Comprehensive model for extracting and updating user profile information.
    
    This model captures all relevant information that can be extracted from
    user messages and updates the persistent user profile.
    """
    
    # Basic Information
    occupation: Optional[str] = Field(
        description="User's occupation in natural language (e.g., 'software developer', 'Freiberufler')"
    )
    occupation_type: Optional[OccupationType] = Field(
        description="Standardized occupation type for tax purposes"
    )
    
    # Personal Status
    marital_status: Optional[MaritalStatus] = Field(
        description="User's marital status"
    )
    tax_class: Optional[TaxClass] = Field(
        description="German tax class (Steuerklasse) if mentioned"
    )
    
    # Dependents
    has_dependents: Optional[bool] = Field(
        description="Whether the user has dependents"
    )
    number_of_children: Optional[int] = Field(
        description="Number of children", ge=0
    )
    dependents: Optional[List[DependentInfo]] = Field(
        description="Detailed information about dependents"
    )
    
    # Income Information
    annual_income: Optional[float] = Field(
        description="Annual income in EUR", ge=0
    )
    income_range: Optional[str] = Field(
        description="Income range category: '0-30k', '30k-60k', '60k-100k', '100k+'"
    )
    income_sources: Optional[List[IncomeSource]] = Field(
        description="Different sources of income"
    )
    
    # Business Information
    is_kleinunternehmer: Optional[bool] = Field(
        description="Whether user is a Kleinunternehmer (small business VAT exemption)"
    )
    business_type: Optional[str] = Field(
        description="Type of business for self-employed users"
    )
    vat_registered: Optional[bool] = Field(
        description="Whether registered for VAT (Umsatzsteuer)"
    )
    
    # Expenses
    known_expenses: Optional[List[str]] = Field(
        description="List of expense descriptions mentioned by user"
    )
    structured_expenses: Optional[List[ExpenseItem]] = Field(
        description="Structured expense items with categories and amounts"
    )
    
    # Additional Context
    tax_year: Optional[int] = Field(
        description="Tax year being discussed", ge=2020, le=2030
    )
    location: Optional[str] = Field(
        description="Location in Germany (relevant for some tax rules)"
    )
    
    # Preferences
    preferred_language: Optional[str] = Field(
        description="Preferred language: 'de' or 'en'"
    )
    risk_tolerance: Optional[str] = Field(
        description="Risk tolerance for tax strategies: 'conservative', 'moderate', 'aggressive'"
    )
    
    @validator('annual_income')
    def validate_income(cls, v):
        if v is not None and v > 10_000_000:  # Reasonable upper limit
            raise ValueError('Income seems unreasonably high')
        return v
    
    @validator('income_range')
    def validate_income_range(cls, v):
        valid_ranges = ['0-30k', '30k-60k', '60k-100k', '100k+']
        if v is not None and v not in valid_ranges:
            raise ValueError(f'Income range must be one of: {valid_ranges}')
        return v


class ProfileUpdateResult(BaseModel):
    """Result of a profile update operation."""
    success: bool = Field(description="Whether the update was successful")
    updates_applied: Dict[str, Any] = Field(description="Fields that were updated")
    validation_errors: List[str] = Field(description="Any validation errors encountered")
    completeness_score: float = Field(description="Updated profile completeness score", ge=0.0, le=1.0)
    suggested_next_questions: List[str] = Field(description="Questions to ask for missing information")


def extract_occupation_type(occupation_text: str) -> Optional[OccupationType]:
    """
    Map natural language occupation to standardized type.
    
    Args:
        occupation_text: User's occupation in natural language
        
    Returns:
        Standardized occupation type or None
    """
    if not occupation_text:
        return None
    
    text = occupation_text.lower()
    
    # Freelancer patterns
    if any(word in text for word in ['freelancer', 'freiberufler', 'selbstständig']):
        if 'freiberufler' in text or 'freelancer' in text:
            return OccupationType.FREELANCER
        return OccupationType.SELF_EMPLOYED
    
    # Employee patterns
    if any(word in text for word in ['angestellt', 'employee', 'mitarbeiter']):
        return OccupationType.EMPLOYEE
    
    # Civil servant patterns
    if any(word in text for word in ['beamter', 'beamtin', 'civil servant']):
        return OccupationType.CIVIL_SERVANT
    
    # Other patterns
    if any(word in text for word in ['rentner', 'pensioner', 'rente']):
        return OccupationType.PENSIONER
    
    if any(word in text for word in ['student', 'studentin']):
        return OccupationType.STUDENT
    
    if any(word in text for word in ['arbeitslos', 'unemployed']):
        return OccupationType.UNEMPLOYED
    
    return OccupationType.OTHER


def categorize_expense(expense_description: str) -> ExpenseCategory:
    """
    Categorize an expense based on its description.
    
    Args:
        expense_description: Natural language expense description
        
    Returns:
        Best matching expense category
    """
    if not expense_description:
        return ExpenseCategory.OTHER
    
    text = expense_description.lower()
    
    # Home office patterns
    if any(word in text for word in ['home office', 'homeoffice', 'büro', 'arbeitszimmer']):
        return ExpenseCategory.HOME_OFFICE
    
    # Commuting patterns
    if any(word in text for word in ['fahrt', 'commute', 'entfernung', 'benzin', 'öpnv']):
        return ExpenseCategory.COMMUTING
    
    # Professional development
    if any(word in text for word in ['fortbildung', 'training', 'kurs', 'seminar', 'weiterbildung']):
        return ExpenseCategory.PROFESSIONAL_DEVELOPMENT
    
    # Work equipment
    if any(word in text for word in ['laptop', 'computer', 'equipment', 'arbeitsmittel']):
        return ExpenseCategory.WORK_EQUIPMENT
    
    # Internet/Phone
    if any(word in text for word in ['internet', 'telefon', 'phone', 'handy']):
        return ExpenseCategory.INTERNET_PHONE
    
    # Software
    if any(word in text for word in ['software', 'abo', 'subscription', 'lizenz']):
        return ExpenseCategory.SOFTWARE_SUBSCRIPTIONS
    
    return ExpenseCategory.OTHER


def validate_profile_updates(updates: ProfileUpdates) -> List[str]:
    """
    Additional validation for profile updates beyond Pydantic validation.
    
    Args:
        updates: Profile updates to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check logical consistency
    if updates.marital_status == MaritalStatus.SINGLE and updates.tax_class in [TaxClass.CLASS_3, TaxClass.CLASS_5]:
        errors.append("Single people cannot have tax class 3 or 5")
    
    if updates.has_dependents is False and updates.number_of_children and updates.number_of_children > 0:
        errors.append("Cannot have children if has_dependents is False")
    
    if updates.is_kleinunternehmer is True and updates.vat_registered is True:
        errors.append("Kleinunternehmer cannot be VAT registered")
    
    # Validate income consistency
    if updates.annual_income and updates.income_range:
        income = updates.annual_income
        range_map = {
            '0-30k': (0, 30000),
            '30k-60k': (30000, 60000), 
            '60k-100k': (60000, 100000),
            '100k+': (100000, float('inf'))
        }
        
        if updates.income_range in range_map:
            min_income, max_income = range_map[updates.income_range]
            if not (min_income <= income <= max_income):
                errors.append(f"Annual income {income} doesn't match range {updates.income_range}")
    
    return errors


def create_expense_suggestions(occupation_type: Optional[OccupationType]) -> List[str]:
    """
    Generate expense suggestions based on occupation type.
    
    Args:
        occupation_type: User's occupation type
        
    Returns:
        List of relevant expense suggestions
    """
    if not occupation_type:
        return []
    
    base_suggestions = [
        "Home Office expenses (Homeoffice-Pauschale)",
        "Professional development courses",
        "Work-related equipment"
    ]
    
    if occupation_type in [OccupationType.FREELANCER, OccupationType.SELF_EMPLOYED]:
        return base_suggestions + [
            "Business internet and phone costs",
            "Software licenses and subscriptions", 
            "Client entertainment expenses",
            "Business travel costs",
            "Professional insurance"
        ]
    
    elif occupation_type == OccupationType.EMPLOYEE:
        return base_suggestions + [
            "Commuting expenses (Entfernungspauschale)",
            "Professional literature",
            "Work clothing and uniforms",
            "Union fees"
        ]
    
    return base_suggestions