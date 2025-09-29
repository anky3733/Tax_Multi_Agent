# agents/knowledge_agent.py
"""
Knowledge Agent - Retrieval-Augmented Generation for German tax questions.

This agent provides factual, grounded answers to tax questions using a RAG pipeline
that retrieves relevant information from a knowledge base and generates personalized responses.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from graph_state import GraphState, UserProfile

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """
    Knowledge Agent responsible for answering tax questions using RAG.
    
    This agent retrieves relevant information from a knowledge base and generates
    personalized, contextual answers based on the user's profile and conversation history.
    """
    
    def __init__(self, retriever: BaseRetriever, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.1):
        """
        Initialize the Knowledge Agent with a retriever and LLM configuration.
        
        Args:
            retriever: Vector store retriever for knowledge base
            model_name: Groq model to use for generation
            temperature: Temperature for response generation (low for factual accuracy)
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._rag_chain = None
    
    @property
    def llm(self):
        """Lazy initialization of the LLM."""
        if self._llm is None:
            try:
                self._llm = ChatGroq(model=self.model_name, temperature=self.temperature)
            except Exception as e:
                logger.error(f"Failed to initialize Knowledge Agent LLM: {e}")
                raise
        return self._llm
    
    def _build_personalized_prompt(self) -> PromptTemplate:
        """
        Build a comprehensive prompt template for personalized tax assistance.
        
        Returns:
            PromptTemplate configured for German tax context with personalization
        """
        template = """You are an expert English speaking tax assistant providing personalized advice based on the user's specific situation.

PERSONALITY & APPROACH:
- Professional but friendly tone
- Clear, actionable explanations
- Use tax terminology with English explanations
- Tailor responses to the user's occupation and circumstances

USER PROFILE CONTEXT:
{user_context}

CONVERSATION CONTEXT:
{conversation_context}

RESPONSE GUIDELINES:

**Structure Rules:**
1.  **Quick Answer**: Start with a single, bolded sentence that directly answers the user's question. Get straight to the point.
2.  **Key Details**: After the Quick Answer, use a bulleted list to provide the most important details (4-6 points max).
3.  **Example**: If applicable, provide a brief, personalized example.
4.  **Closing**: Conclude naturally. You do not need to ask a follow-up question unless it is highly relevant.

**Content Rules:**
- Base answers ONLY on the provided context documents.
- If information is not in the context, clearly state this.
- Prioritize information most relevant to the user's profile.

**Formatting Rules:**
- Use clear Markdown.
- Highlight key numbers and terms: **€920**, **Werbungskostenpauschale**.

---
**EXAMPLE RESPONSE STRUCTURE:**

**Quick Answer:** Yes, for low-value assets, there's a limit of **€800 (net)** for immediate full deduction in the year of purchase.

**Key Details:**
*   This applies to "Geringwertige Wirtschaftsgüter" (GWG), or low-value assets.
*   Your **€1,200 laptop** is above this limit, so it must be depreciated over its official useful life (usually 3 years for computers).
*   For your **€250 office chair**, since it's below the €800 limit, you can deduct the full amount immediately.

Would you like me to explain how depreciation for the laptop works?
---

KNOWLEDGE BASE CONTEXT:
{context}

USER PROFILE CONTEXT:
{user_context}

CONVERSATION CONTEXT:
{conversation_context}

QUESTION:
{question}

Provide a concise, scannable, and personalized answer following the structure rules above."""

        return PromptTemplate.from_template(template)
    
    def _extract_user_context(self, user_profile: UserProfile) -> str:
        """
        Extract relevant context from user profile for personalization.
        
        Args:
            user_profile: User's profile information
            
        Returns:
            Formatted string with relevant user context
        """
        context_parts = []
        
        # Occupation and employment type
        if user_profile.get("occupation"):
            context_parts.append(f"Occupation: {user_profile['occupation']}")
        
        if user_profile.get("occupation_type"):
            context_parts.append(f"Employment type: {user_profile['occupation_type']}")
        
        # Personal status
        if user_profile.get("marital_status"):
            context_parts.append(f"Marital status: {user_profile['marital_status']}")
            if user_profile.get("spouse_annual_income") is not None: # <-- ADD THIS BLOCK
                spouse_income = user_profile['spouse_annual_income']
                if spouse_income == 0:
                    context_parts.append("Spouse's status: Not working (income €0)")
                else:
                    context_parts.append(f"Spouse's annual income: €{spouse_income:,.0f}")
        
        if user_profile.get("tax_class"):
            context_parts.append(f"Tax class: {user_profile['tax_class']}")
        
        if user_profile.get("has_dependents"):
            children = user_profile.get("number_of_children", 0)
            context_parts.append(f"Has dependents: Yes ({children} children)" if children else "Has dependents: Yes")
        
        # Income information
        if user_profile.get("annual_income"):
            context_parts.append(f"Annual income: €{user_profile['annual_income']:,.0f}")
        elif user_profile.get("income_range"):
            context_parts.append(f"Income range: {user_profile['income_range']}")
        
        # Business information
        if user_profile.get("is_kleinunternehmer"):
            context_parts.append("Status: Kleinunternehmer (small business)")
        
        if user_profile.get("vat_registered"):
            context_parts.append("VAT registered: Yes")
        
        # Known expenses for context
        expenses = user_profile.get("known_expenses", [])
        if expenses:
            context_parts.append(f"Current expenses: {', '.join(expenses[:3])}{'...' if len(expenses) > 3 else ''}")
        
        return "\n".join(context_parts) if context_parts else "No specific profile information available."
    
    def _extract_conversation_context(self, messages: List[BaseMessage], max_messages: int = 5) -> str:
        """
        Extract recent conversation context for better question understanding.
        
        Args:
            messages: Conversation message history
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        if not messages:
            return "No previous conversation context."
        
        # Get the last few messages for context
        recent_messages = messages[-max_messages:]
        
        context_lines = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                context_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Truncate long AI responses to keep context manageable
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_lines.append(f"Assistant: {content}")
        
        return "\n".join(context_lines)
    
    def _enhance_retrieval_query(self, question: str, user_profile: UserProfile) -> str:
        """
        Enhance the retrieval query with user context for better document matching.
        
        Args:
            question: Original user question
            user_profile: User profile for context
            
        Returns:
            Enhanced query string
        """
        query_parts = [question]
        
        # Add occupation context for better retrieval
        if user_profile.get("occupation_type"):
            occupation_type = user_profile["occupation_type"]
            if occupation_type in ["freelancer", "self_employed"]:
                query_parts.append("Freiberufler selbstständig")
            elif occupation_type == "employee":
                query_parts.append("Angestellter Arbeitnehmer")
        
        # Add marital status for relevant queries
        if "married" in question.lower() or user_profile.get("marital_status") == "married":
            query_parts.append("verheiratet Ehegatte")
        
        # Add business context
        if user_profile.get("is_kleinunternehmer"):
            query_parts.append("Kleinunternehmer")
        
        return " ".join(query_parts)
    
    def _build_rag_chain(self):
        """Build the complete RAG chain with personalization."""
        if self._rag_chain is None:
            prompt = self._build_personalized_prompt()
            
            self._rag_chain = (
                {
                    "context": lambda x: self._retrieve_documents(x["enhanced_query"]),
                    "question": lambda x: x["question"],
                    "user_context": lambda x: x["user_context"],
                    "conversation_context": lambda x: x["conversation_context"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        
        return self._rag_chain
    
    def _retrieve_documents(self, query: str) -> str:
        """
        Retrieve and format documents from the knowledge base.
        
        Args:
            query: Search query
            
        Returns:
            Formatted document context
        """
        try:
            documents = self.retriever.invoke(query)
            if not documents:
                return "No relevant information found in the knowledge base."
            
            # Format documents with source information
            formatted_docs = []
            for i, doc in enumerate(documents[:5]):  # Limit to top 5 documents
                content = doc.page_content
                source = getattr(doc, 'metadata', {}).get('source', f'Document {i+1}')
                formatted_docs.append(f"[Source: {source}]\n{content}")
            
            return "\n\n".join(formatted_docs)
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return "Error retrieving information from knowledge base."
    
    async def answer_question(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a personalized answer to the user's question using RAG.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dictionary with the generated response and metadata
        """
        logger.info("--- Running Knowledge Agent ---")
        
        try:
            # Validate input
            messages = state.get("messages", [])
            if not messages:
                logger.warning("No messages found in state")
                return {
                    "direct_response": "I don't see any question to answer. Please ask me about German taxes!",
                    "response_metadata": {
                        "response_type": "error",
                        "confidence_score": 0.0,
                        "sources_used": []
                    }
                }
            
            # Extract the current question
            last_message = messages[-1]
            if not hasattr(last_message, 'content') or not last_message.content:
                logger.warning("Empty or invalid last message")
                return {
                    "direct_response": "I didn't receive a clear question. Could you please rephrase your tax question?",
                    "response_metadata": {
                        "response_type": "error",
                        "confidence_score": 0.0,
                        "sources_used": []
                    }
                }
            
            question = last_message.content
            user_profile = state.get("user_profile", {})
            
            # Build context for personalization
            user_context = self._extract_user_context(user_profile)
            conversation_context = self._extract_conversation_context(messages)
            enhanced_query = self._enhance_retrieval_query(question, user_profile)
            
            logger.info(f"Processing question: {question}")
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Build and execute RAG chain
            rag_chain = self._build_rag_chain()
            
            response = await rag_chain.ainvoke({
                "question": question,
                "enhanced_query": enhanced_query,
                "user_context": user_context,
                "conversation_context": conversation_context
            })
            
            logger.info("Successfully generated response")
            
            # Track personalization applied
            personalization_applied = []
            if user_profile.get("occupation"):
                personalization_applied.append("occupation_specific")
            if user_profile.get("marital_status"):
                personalization_applied.append("marital_status")
            if user_profile.get("known_expenses"):
                personalization_applied.append("expense_history")
            
            return {
                "direct_response": response,
                "response_metadata": {
                    "response_type": "answer",
                    "confidence_score": 0.8,  # Could be calculated based on retrieval scores
                    "sources_used": ["tax_knowledge_base"],
                    "personalization_applied": personalization_applied,
                    "reasoning_steps": [
                        "Retrieved relevant tax information",
                        "Applied user profile context", 
                        "Generated personalized response"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Knowledge Agent: {e}")
            return {
                "direct_response": "I apologize, but I encountered an error while processing your question. Please try asking again or rephrase your question.",
                "response_metadata": {
                    "response_type": "error",
                    "confidence_score": 0.0,
                    "sources_used": [],
                    "error_message": str(e)
                }
            }


# Global instance and backward compatibility
_knowledge_agent_instance = None


def create_knowledge_agent(retriever: BaseRetriever, model_name: str = "openai/gpt-oss-120b") -> KnowledgeAgent:
    """
    Create a Knowledge Agent instance (backward compatible function).
    
    Args:
        retriever: Vector store retriever
        model_name: Groq model name
        
    Returns:
        Configured KnowledgeAgent instance
    """
    global _knowledge_agent_instance
    _knowledge_agent_instance = KnowledgeAgent(retriever, model_name)
    return _knowledge_agent_instance


async def run_knowledge_agent(state: GraphState, rag_chain=None) -> Dict[str, Any]:
    """
    Main entry point for knowledge agent (backward compatible).
    
    Args:
        state: Graph state
        rag_chain: Legacy parameter (now ignored)
        
    Returns:
        Response dictionary
    """
    if _knowledge_agent_instance is None:
        logger.error("Knowledge agent not initialized. Call create_knowledge_agent first.")
        return {
            "direct_response": "Knowledge agent not properly initialized.",
            "response_metadata": {
                "response_type": "error",
                "confidence_score": 0.0
            }
        }
    
    return await _knowledge_agent_instance.answer_question(state)


def get_knowledge_agent_instance() -> Optional[KnowledgeAgent]:
    """Get the global knowledge agent instance."""
    return _knowledge_agent_instance