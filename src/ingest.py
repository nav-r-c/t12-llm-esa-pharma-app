import json
from langchain.schema import Document  # Import the Document class
from langchain.vectorstores import FAISS
from langchain.embeddings.ollama import OllamaEmbeddings  

# Load dataset and convert to Document objects
def load_dataset(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                # Convert each dictionary to a Document object
                documents.append(Document(page_content=value, metadata={"section": key}))
    return documents

# Create vector store from Document objects
def create_vector_store(documents, embedding_model="llama3.2"):
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Main script
if __name__ == "__main__":
    import os 

    data_directory = "C:\\Users\\abhin\\Documents\\Code\\pharma-assistant\\data\\"
    file_paths = [
        "C:\\Users\\abhin\\Documents\\Code\\pharma-assistant\\data\\Acetazolamide Extended-Release Capsules.json",
        "C:\\Users\\abhin\\Documents\\Code\\pharma-assistant\\data\\Rasagiline Tablets.json",
    ]
    # file_paths = ['C:\\Users\\abhin\\Documents\\Code\\pharma-assistant\\data\\Acetazolamide Extended-Release Capsules.json']
    print(file_paths)


    documents = load_dataset(file_paths)
    print(f"Loaded {len(documents)} documents.")
    vector_store = create_vector_store(documents)
    vector_store.save_local("vector_store")
    print("Vector store saved successfully.")
