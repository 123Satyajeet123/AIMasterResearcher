# tools/search_tool.py
import os
import json
import requests
from dotenv import load_dotenv
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

load_dotenv()
serp_api_key = os.getenv("SERP_API_KEY")

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to be executed")

def search(query: str):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': serp_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, data=payload)
    results = json.loads(response.text)
    
    # Extract only the organic results
    organic_results = results.get('organic', [])
    
    # Format the results
    formatted_results = []
    for result in organic_results:
        formatted_results.append({
            'title': result.get('title'),
            'link': result.get('link'),
            'snippet': result.get('snippet')
        })
    
    # print(formatted_results)
    return json.dumps(formatted_results)


# result = search("Carbon markets trends")

# print(result)

search_tool = Tool(
    name="Search",
    func=search,
    description="Searches the internet for relevant links based on a given query.",
    args_schema=SearchInput
)