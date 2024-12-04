from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings  
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tavily import TavilyClient

def call_external_api(query):
    print("External API called.")
    client = TavilyClient(api_key="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")

    response = client.search(query=query)

    return response['results'][0]['content']    

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
        
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        
        # Extract content from retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])
        
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

def get_answer(query):
    # Load the vector store with Ollama embeddings
    vector_store = FAISS.load_local("vector_store", OllamaEmbeddings(model="llama3.2"), allow_dangerous_deserialization=True)
    
    # Retrieve documents using vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Define the prompt template
    prompt_template = """You are a helpful medical assistant.
    Use the following context to answer the question: 
    Context: {documents}
    Question: {question}
    Answer the question based on the context provided. 
    If the context does not contain sufficient information to answer the question, respond with 'I don't know'."""

    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "documents"]
    )

    # Initialize the language model (ChatOllama)
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Create the rag_chain using the prompt and LLM
    rag_chain = prompt | llm | StrOutputParser()

    # Create the RAGApplication
    rag_application = RAGApplication(retriever, rag_chain)

    # Get the result from RAG
    result = rag_application.run(query)
    
    # Clean the result (if necessary, based on the output format)
    result = result.strip()

    # If the result is 'unknown', call the external API to fetch more information
    if "I don't know" in result:
        # Call the external API for additional results
        api_results = call_external_api(query)
        result = "\n\nHere are some web results:\n"
        result += "".join(api_results)
    
    return result

# Example usage
print(get_answer("Can you tell me what minecraft is?"))
