from tavily import TavilyClient

def call_external_api(query):
    client = TavilyClient(api_key="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")

    response = client.search(query=query)

    return response['results'][0]['content']    