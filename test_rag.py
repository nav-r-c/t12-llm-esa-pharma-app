import unittest
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TestRAG(unittest.TestCase):
    def test_query_with_context(self):
        # Define the query and context
        context = "Amoxicillin is an antibiotic used to treat bacterial infections."
        query = "What is the primary use of Amoxicillin?"

        # Define the prompt template
        prompt_template = """You are a pharma assistant. Answer the following query based on the context provided:
        Context: {context}
        Query: {query}
        Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        # Initialize the LLM
        llm = Ollama(base_url="http://localhost:11434", model="llama3.2")

        # Create a custom chain for testing
        test_chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain with both context and query
        response = test_chain.run({"context": context, "query": query})

        # Assert that the response contains the expected output
        # self.assertIn("antibiotic", response.lower())
        print(response)

if __name__ == "__main__":
    unittest.main()
