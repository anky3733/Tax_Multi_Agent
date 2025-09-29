# Comprehensive Test Cases for Taxfix Multi-Agent System

## Test Category 1: Profile Building & Memory Integration

### Test 1: Complex Freelancer Profile Building
**Input:** "Hi, I'm a freelance software consultant. I work from home about 3 days a week, earn around 4500€ per month, and I'm married with two children. Last year I bought a new laptop for 2800€ and spent about 1200€ on business travel."

**Expected System Behavior:**
- **Router Agent:** Routes to profile_manager 
- **Profile Manager:** Extracts occupation=Freiberufler, monthly_income=4500€, annual_income=54000€, marital_status=married, has_dependents=true, known_expenses=[laptop purchase, business travel, home office]
- **Action Proposer:** Suggests tax class optimization or VAT status check
- **Memory:** Profile persists across conversation

**Expected Answer:** Should acknowledge profile update, provide brief overview of relevant deductions (home office, equipment, travel), and suggest next step like tax class optimization.

---

### Test 2: Incomplete Profile with Follow-up
**Input Session:**
1. "I'm a freelancer"
2. "I work in graphic design" 
3. "I earn about 3000€ monthly"

**Expected System Behavior:**
- Each message should update profile incrementally
- System should remember previous information
- Action Proposer should suggest adding expenses after income is provided
- **Memory Test:** Profile should show occupation=Freiberufler, monthly_income=3000€, profession details

**Expected Answer:** Progressive profile building with contextual suggestions becoming more specific as profile completes.

---

## Test Category 2: Knowledge Retrieval & Personalization

### Test 3: Personalized Home Office Deduction
**Input:** "How does the home office deduction work for me?" 
**Context:** User profile shows Freiberufler, works from home, 3500€ monthly income

**Expected System Behavior:**
- **Router Agent:** Routes to knowledge_agent
- **Knowledge Agent:** Uses RAG to find home office rules, personalizes based on freelancer status
- **Action Proposer:** Suggests tracking home office days

**Expected Answer:** Should explain 6€/day rule, 210-day limit, mention it's particularly beneficial for freelancers, and ask about current home office setup or suggest starting to track days.

---

### Test 4: Complex Deduction Scenario
**Input:** "I bought a €3500 MacBook Pro for work, a €800 monitor, and €200 worth of design software. What can I deduct?"

**Expected System Behavior:**
- **Knowledge Agent:** Should distinguish between different depreciation rules
- Should update known_expenses in profile
- **Action Proposer:** Might suggest adding more equipment expenses

**Expected Answer:** MacBook can be fully depreciated in first year (special computer rule), monitor under €800 threshold so full deduction, software as business expense. Should ask if equipment is used >90% for business.

---

### Test 5: VAT/Kleinunternehmer Decision
**Input:** "I'm a new freelancer expecting to earn about 35000€ this year. Should I register for VAT or use Kleinunternehmer?"

**Expected System Behavior:**
- **Knowledge Agent:** Should access VAT rules from knowledge base
- Should consider user's income level (above 22k threshold)
- **Profile Manager:** Should note income and VAT question
- **Action Proposer:** Might suggest calculating input VAT on business purchases

**Expected Answer:** Above €22k so not automatically eligible for Kleinunternehmer, but can still choose it if under €50k. Should explain trade-offs: VAT collection vs. input VAT recovery, and ask about planned business purchases.

---

## Test Category 3: Multi-Turn Conversational Flow

### Test 6: Complex Tax Planning Conversation
**Input Sequence:**
1. "I'm married and my wife doesn't work. We have one child."
2. "I earn 65000€ annually as an employee"
3. "What's the best tax class for us?"
4. "What other deductions should I consider?"

**Expected System Behavior:**
- Each turn should build on previous information
- Should maintain conversation context
- **Memory Integration:** Tax class advice should consider income level and family situation
- **Action Proposer:** Should suggest family-related deductions

**Expected Answer Flow:** 
1. Profile acknowledgment, suggest tax class discussion
2. Recommend tax class III/V combination, explain benefits
3. Suggest child allowance, potential home office if applicable, commuting deductions

---

### Test 7: Error Correction and Context Maintenance
**Input Sequence:**
1. "I'm single and earn 45000€"
2. "Actually, I'm married, sorry for the confusion"
3. "Can you recalculate my tax situation?"

**Expected System Behavior:**
- **Profile Manager:** Should update marital_status from single to married
- Should maintain income information
- **Knowledge Agent:** Should provide updated advice based on corrected profile
- **Memory:** Should show corrected profile state

**Expected Answer:** Should acknowledge correction, update recommendations based on married status, suggest tax class optimization.

---

## Test Category 4: Action-Oriented Interactions

### Test 8: Expense Addition Flow
**Input:** "As a freelancer, what expenses should I track?"
**Follow-up:** User clicks "Add Expenses" action
**Response:** "Office rent 800€ monthly, internet 50€ monthly, Adobe subscription 60€ monthly"

**Expected System Behavior:**
- Initial question routes to knowledge_agent
- **Action Proposer:** Should suggest adding specific expenses
- UI should show appropriate action button
- **Profile Manager:** Should process expense list and add to known_expenses
- Should provide confirmation of added expenses

**Expected Answer:** Initial response about common freelancer expenses, then action suggestion, then confirmation of specific expenses added.

---

### Test 9: Income Information Flow
**Input:** "I just started freelancing"
**Action:** User clicks "Add Income Info" 
**Response:** "I expect to earn around 4000€ per month"

**Expected System Behavior:**
- **Action Proposer:** Should detect new freelancer without income info
- Should suggest adding income with appropriate action_type
- **Profile Manager:** Should extract and validate income information
- Should confirm income addition and suggest next logical step

**Expected Answer:** Welcome new freelancer, income confirmation, suggest expense tracking or VAT registration discussion.

---

## Test Category 5: Edge Cases & Error Handling

### Test 10: Ambiguous Information
**Input:** "I sometimes work from home and sometimes at client offices. I have some business expenses but I'm not sure what's deductible."

**Expected System Behavior:**
- Should handle uncertainty gracefully
- **Knowledge Agent:** Should provide general guidance and ask clarifying questions
- **Action Proposer:** Should suggest specific information gathering

**Expected Answer:** Explain mixed-use rules, ask for specifics about home office frequency and expense types, suggest expense categorization.

---

### Test 11: Contradictory Information
**Input:** "I'm a Kleinunternehmer earning 60000€ annually"

**Expected System Behavior:**
- Should identify contradiction (Kleinunternehmer income limit is €50k)
- **Knowledge Agent:** Should explain the inconsistency
- **Profile Manager:** Should note the conflicting information
- Should ask for clarification

**Expected Answer:** Point out income exceeds Kleinunternehmer limit, explain VAT implications, ask for clarification on current status.

---

### Test 12: Complex Family Situation
**Input:** "I'm divorced, have two children who live with me half the time, work as both an employee (30 hours) and freelancer (10 hours), and my ex-spouse and I alternate claiming child benefits each year."

**Expected System Behavior:**
- **Profile Manager:** Should handle complex marital/dependent status
- **Knowledge Agent:** Should address mixed employment types
- Should identify multiple complex tax implications
- **Action Proposer:** Should suggest prioritizing most impactful optimizations

**Expected Answer:** Acknowledge complex situation, address mixed income types, explain child benefit alternation rules, suggest focusing on employment vs. freelance deduction optimization.

---

## Test Category 6: Streaming & Transparency Features

### Test 13: Complex Research Query
**Input:** "Explain the complete ELSTER filing process for a freelancer with international clients"

**Expected System Behavior:**
- Should show streaming status updates (Router → Knowledge Agent → Action Proposer)
- **Transparent Reasoning:** Should show which knowledge base sections are being accessed
- Response should stream in chunks, not appear all at once
- **Action Proposer:** Should suggest practical next steps

**Expected Answer:** Step-by-step ELSTER process with international considerations, streamed progressively with visible agent reasoning.

---

### Test 14: Memory Impact Demonstration
**Input:** "What deductions apply to my situation?"

**Expected System Behavior:**
- Should demonstrate how user profile influences response
- **Transparent Memory Usage:** Should show which profile elements are being used
- Response should be highly personalized based on accumulated profile data
- Sidebar should show profile utilization

**Expected Answer:** Highly personalized deduction list based on accumulated profile (occupation, income, family status, known expenses), with clear reasoning about why each is relevant.

---

## Test Category 7: German Tax Specificity

### Test 15: Advanced German Tax Scenario
**Input:** "I'm a Freiberufler doctor with my own practice, married to a Beamter, we have three children, I earn 120000€ annually, my spouse earns 55000€. We're considering buying a practice building. What's our optimal tax strategy?"

**Expected System Behavior:**
- **Profile Manager:** Should handle complex professional classifications
- **Knowledge Agent:** Should access advanced tax rules for medical professionals
- Should consider Beamter spouse implications
- **Action Proposer:** Should suggest consulting tax advisor for investment decisions

**Expected Answer:** Address professional status differences, tax class optimization, child benefits, investment implications for practice building, recommend professional consultation for complex real estate decisions.

---

## Success Criteria for Each Test:

### 1. **Memory Integration (40% weight)**
- Profile information persists across conversation turns
- New information correctly updates existing profile
- Responses become more personalized as profile builds
- Contradictions are identified and resolved

### 2. **Agent Coordination (25% weight)**
- Proper routing between agents
- Knowledge agent provides accurate, RAG-based answers
- Action proposer suggests contextually appropriate actions
- Profile manager correctly extracts and validates information

### 3. **User Experience (20% weight)**
- Streaming responses work smoothly
- Transparent reasoning is displayed
- Action buttons match proposed actions
- Conversation flows naturally

### 4. **German Tax Accuracy (15% weight)**
- Factually correct tax information
- Proper German tax terminology
- Accurate calculations and thresholds
- Appropriate complexity for user level

## Performance Benchmarks:
- **Response Time:** <3 seconds for simple queries, <8 seconds for complex ones
- **Profile Accuracy:** >95% correct information extraction
- **Knowledge Accuracy:** >90% factually correct tax information
- **Action Relevance:** >85% of proposed actions should be appropriate for user context