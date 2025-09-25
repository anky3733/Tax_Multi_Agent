# agents/knowledge_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

def create_knowledge_agent(retriever):
    """
    Creates the RAG chain for the knowledge agent.
    """
    
    template = """You are a helpful tax assistant. Answer the user's question based only on the following context.
    If the context doesn't contain the answer, state that you don't have enough information.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    # This is the RAG chain
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

async def run_knowledge_agent(state: dict):
    """
    The node for the knowledge agent in the graph.
    """
    print("--- Running Knowledge Agent ---")
    question = state["messages"][-1].content
    rag_chain = state['rag_chain'] # Assumes rag_chain is passed in the state
    
    # Invoke the RAG chain with the latest question
    response = await rag_chain.ainvoke({"question": question})
    
    # We will append the response to the messages list in a later step
    return {"response_from_knowledge_agent": response}