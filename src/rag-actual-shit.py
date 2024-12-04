from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings  
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

def get_answer(query):

    vector_store = FAISS.load_local("vector_store", OllamaEmbeddings(model="llama3.2"), allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})


    prompt = """You are a helpful medical assistant. 
    Use the following context to answer the question:Context:{documents} Question: {question} 
    Provide a clear, concise, and accurate answer based on the given context. 
    If the information is not in the context, say "I don't have enough specific information to answer this question."""

    prompt = PromptTemplate(
        template=prompt,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    rag_chain = prompt | llm | StrOutputParser()

    rag_application = RAGApplication(retriever, rag_chain)

    result = rag_application.run(query)
    return result
