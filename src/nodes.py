from utils import call_external_api
from langchain.prompts import PromptTemplate

class Summarize:
    def __init__(self, inputs, retriever):
        self.query = inputs['query']
        self.retrieved_docs = inputs['retrieved_docs']  # The retrieved documents
        self.retriever = retriever

        print(self.query)
        print(self.retrieved_docs)

    def process(self):
        query = self.query
        context = "\n\n".join([doc.page_content for doc in self.retrieved_docs])
        
        prompt = """You are a pharma expert. Summarize the details about the following drug:
        Context: {context}
        Query: {query}
        Summary: """.format(context=context, query=query)

        prompt = PromptTemplate(template=prompt)
        
        # Pass the generated prompt to the retriever (rag_chain)
        return self.retriever.invoke({"question": query, "documents": prompt})


class Recommendation:
    def __init__(self, inputs, retriever):
        self.query = inputs['query']
        self.retrieved_docs = inputs['retrieved_docs']  # The retrieved documents
        self.retriever = retriever


    def process(self):
        query = self.query
        context = "\n\n".join([doc.page_content for doc in self.retrieved_docs])

        prompt = """You are a pharma expert. Based on the user's condition, suggest an alternative medicine or recommendation:
        Context: {context}
        Query: {query}
        Recommendation: """.format(context=context, query=query)

        prompt = PromptTemplate(template=prompt)
        
        # Pass the generated prompt to the retriever (rag_chain)
        return self.retriever.invoke({"question": query, "documents": prompt})


class UnknownResponse:
    def __init__(self, inputs, retriever):
        self.query = inputs['query']
        self.retrieved_docs = inputs['retrieved_docs']  # The retrieved documents
        self.retriever = retriever


    def process(self):
        query = self.query
        context = "\n\n".join([doc.page_content for doc in self.retrieved_docs])
        
        prompt = """You are a pharma expert. If you don't have enough information, respond with 'I don't know, but here are some web results:'. 
        Context: {context}
        Query: {query}
        Response: """.format(context=context, query=query)

        prompt = PromptTemplate(template=prompt)

        response = self.retriever.invoke({"question": query, "documents": prompt})

        if "I don't know" in response:
            # Call an external API to fetch web results if needed
            api_results = call_external_api(query)
            response += "\n\nHere are some web results:\n"
            response += "\n".join([result['title'] for result in api_results.get("results", [])])
        return response
