from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph
from langchain.agents import AgentType as AgentState
import ollama

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=2000)
docs = ["Your text documents here..."]  # Replace with your actual documents
splits = text_splitter.create_documents(docs)
print(f"Documents split into {len(splits)} chunks")

# Create embeddings and vector store
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Verify embeddings work
    test_embedding = embeddings.embed_documents(["Test document"])
    print("Embedding generation successful!")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("Vector store created successfully!")

    # Define retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Error creating vector store: {e}")

# Node 1: RAG (Retrieval-Augmented Generation) Node for answering questions
def rag_node(query):
    retrieved_docs = retriever.invoke(query)

    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Construct prompt with context and query
    prompt = f"""You are a helpful medical assistant. Use the following context to answer the question:

Context:
{context}

Question: {query}

Provide a clear, concise, and accurate answer based on the given context. If the information is not in the context, say "I don't have enough specific information to answer this question."
"""

    # Generate response using Ollama/Llama 3
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])

    return response['message']['content'], retrieved_docs

# Node 2: Summary Node for summarizing retrieved documents
def summary_node(query):
    retrieved_docs = retriever.invoke(query)

    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Construct prompt with context and query
    prompt = f"""You are a helpful medical assistant. Use the following context to summarize:

Context:
{context}

Question: {query}

Provide a clear, concise, and accurate summary based on the given context. If the information is not in the context, say "I don't have enough specific information to summarize this content."
"""

    # Generate response using Ollama/Llama 3
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])

    return response['message']['content'], retrieved_docs

# Create a StateGraph for agentic flow
builder = StateGraph(AgentState)

# Add nodes for retrieval and summarization
builder.add_node("rag", rag_node)
builder.add_node("summary", summary_node)

# Set the entry point to the RAG node
builder.set_entry_point("rag")

# Add conditional transitions based on task type
# Assuming a task condition to switch to "summary" based on the query (example: query contains 'summarize')
def get_next_task(query):
    if 'summarize' in query.lower():
        return "summary"
    else:
        return "rag"

# Adding edges for transitions based on query type
builder.add_edge("rag", "summary", condition=lambda query: 'summarize' in query.lower())  # Transition to summary if query contains 'summarize'
builder.add_edge("summary", "rag", condition=lambda query: 'summarize' not in query.lower())  # Return to RAG if it's not a summarization task

# Compile the graph with checkpointer memory (this could be a simple in-memory object for demonstration)
memory = {}  # Use an in-memory store or database for tracking state (if necessary)
graph = builder.compile(checkpointer=memory)

# Example usage
query = "Can you summarize the following medical notes?"
task_type = get_next_task(query)

# Process the query through the graph, based on the determined task type
current_node = task_type
response, docs = graph.run(query, current_node=current_node)

print("Response:", response)
print("Retrieved documents:", [doc.page_content for doc in docs])