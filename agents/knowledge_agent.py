# agents/knowledge_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

def create_knowledge_agent(retriever):
    """
    Creates the RAG chain for the knowledge agent.
    """
    
    template = """You are a helpful tax assistant. Answer the user's question based only on the following context.
    If the context doesn't contain the answer, state that you don't have enough information.

        --- FORMATTING RULES ---
    - You MUST use Markdown formatting.
    - Keep paragraphs short and focused.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # CHANGE THE LLM INITIALIZATION
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
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
    
    # --- NEW CONTEXT-AWARE LOGIC ---
    # Get the last 5 messages to provide context
    recent_messages = state["messages"][-5:]
    
    # Format the messages into a single string
    contextual_question = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    
    print(f"--- Contextual Question for RAG: ---\n{contextual_question}\n---------------------------------")
    
    # Invoke the RAG chain with the full context
    response = await rag_chain.ainvoke({"question": contextual_question})
    
    return {"direct_response": response}