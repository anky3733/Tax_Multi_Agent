async def run_expense_handler(state: dict):
    """
    Handles the user's request to add an expense by prompting for details.
    """
    print("--- Running Expense Handler ---")
    
    prompt_message = "Of course. What is the expense you would like to add? Please describe it and provide the amount (e.g., 'Laptop purchase, 1200 USD')."
    
    return {"direct_response": prompt_message}