# agents/knowledge_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

def create_knowledge_agent(retriever):
    """
    Creates the RAG chain for the knowledge agent, contextualized for the German tax system.
    """
    
    # --- UPDATED GERMAN-CONTEXT PROMPT ---
    template = """You are a helpful tax assistant specializing in the German tax system. Your goal is to provide clear, accurate answers based ONLY on the provided context.
Use German tax terminology where appropriate (e.g., Werbungskosten, Home-Office-Pauschale, Freiberufler) but explain it in English.

--- CONTENT RULES ---
- You MUST NOT add a top-level title or introduction that is not present in the context.
- Start the answer directly by addressing the user's question.

--- FORMATTING RULES ---
- You MUST use Markdown formatting.
- Use third-level headings (`###`) for main topics or sections.
- Use bullet points (`*` or `-`) for lists of items.
- Use bold text (`**text**`) to highlight key terms, numbers, and concepts (e.g., **6 € per day**, **1,260 €**).
- Keep paragraphs short and focused.

--- EXAMPLE OF CORRECT FORMATTING ---
**Eligibility**
* The space must be used exclusively and regularly for your trade or business.

**Annual Limit**
* The deduction is capped at a maximum of **210 home-office days** per year.
---

---
Context:
{context}

Question:
{question}
"""
    
    prompt = PromptTemplate.from_template(template)
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Or another capable model
    
    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]), 
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


async def run_knowledge_agent(state: dict, rag_chain):
    """
    The node for the knowledge agent in the graph.
    This version includes recent history for better context.
    """
    print("--- Running Knowledge Agent ---")
    
    recent_messages = state["messages"][-5:]
    
    contextual_question = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    
    print(f"--- Contextual Question for RAG: ---\n{contextual_question}\n---------------------------------")
    
    response = await rag_chain.ainvoke({"question": contextual_question})
    
    return {"direct_response": response}