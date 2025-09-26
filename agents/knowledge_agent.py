# agents/knowledge_agent.py

# This agent's sole responsibility is to answer user questions based on a provided knowledge base.
# It uses a Retrieval-Augmented Generation (RAG) pipeline to ensure answers are factual and grounded.

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

def create_knowledge_agent(retriever):
    """
    This function builds the core logic of the Knowledge Agent.
    It constructs a LangChain Expression Language (LCEL) chain that performs RAG.

    Args:
        retriever: An object capable of fetching relevant documents from a vector store
                   based on a query.

    Returns:
        A runnable LangChain chain.
    """
    
    # This prompt template is the "brain" of the agent. It defines its persona, rules, and how it should format its response.
    # Giving it a persona ("specializing in the German tax system") and explicit rules makes the output more reliable.
    template = """You are a helpful tax assistant specializing in the German tax system. Your goal is to provide clear, accurate answers based ONLY on the provided context.
Use German tax terminology where appropriate (e.g., Werbungskosten, Home-Office-Pauschale, Freiberufler) but explain it in English.

--- CONTENT RULES ---
# These rules prevent the model from hallucinating or providing unhelpful introductions.
- You MUST NOT add a top-level title or introduction that is not present in the context.
- Start the answer directly by addressing the user's question.

--- FORMATTING RULES ---
# These rules ensure the user experience is clean and readable in the Streamlit UI.
- You MUST use Markdown formatting.
- Use **bold text on its own line** for main topics (e.g., **Eligibility**).
- Beneath each bolded topic, use bullet points (`*` or `-`) for the details.
- **Crucially: Do NOT put a bullet point on the main topic line itself.**
- Use bold text (`**...**`) within sentences to highlight key numbers and terms like **6 € per day** or **ELSTER**.

--- EXAMPLE OF CORRECT FORMATTING ---
# This "few-shot" example gives the LLM a perfect pattern to copy, drastically improving formatting consistency.
**Eligibility**
* The space must be used exclusively and regularly for your trade or business.

**Annual Limit**
* The deduction is capped at a maximum of **210 home-office days** per year.
---

Context:
{context}

Question:
{question}
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Initialize the Large Language Model. temperature=0 makes the output deterministic and factual, which is crucial for a tax assistant.
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) # Or another capable model
    
    # This is the RAG chain, built using LangChain's pipe syntax (`|`).
    # It defines a sequence of operations to process the user's input.
    rag_chain = (
        {
            # This dictionary runs two tasks in parallel:
            # 1. "context": Fetches relevant documents from the vector store using the retriever.
            # 2. "question": Passes the original user question through.
            "context": lambda x: retriever.invoke(x["question"]), 
            "question": lambda x: x["question"]
        }
        | prompt  # The dictionary output is "piped" into the prompt, filling in the {context} and {question} variables.
        | llm     # The populated prompt is sent to the LLM to generate an answer.
        | StrOutputParser() # This ensures the final output from the LLM is a clean string.
    )
    
    return rag_chain


async def run_knowledge_agent(state: dict, rag_chain):
    """
    This is the entry point function for the Knowledge Agent when called by the LangGraph orchestrator.
    It takes the current conversation state and the RAG chain, executes the chain, and returns the result.

    Args:
        state: The current state of the graph, containing messages and user profile.
        rag_chain: The pre-built RAG chain from the create_knowledge_agent function.

    Returns:
        A dictionary with the key 'direct_response', which updates the main graph's state.
    """
    print("--- Running Knowledge Agent ---")
    
    # To handle ambiguous questions like "Is there a limit?", we provide the last few messages as context.
    # This gives the agent short-term memory of the immediate conversation.
    recent_messages = state["messages"][-5:]
    
    # Format the recent history into a single string for the RAG chain.
    contextual_question = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])
    
    print(f"--- Contextual Question for RAG: ---\n{contextual_question}\n---------------------------------")
    
    # Execute the RAG chain with the full conversational context.
    response = await rag_chain.ainvoke({"question": contextual_question})
    
    # The return value is a dictionary. LangGraph will use this to update the 'direct_response'
    # field in the central `GraphState`.
    return {"direct_response": response}