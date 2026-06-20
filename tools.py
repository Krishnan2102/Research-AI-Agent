from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from typing import Dict,Any

load_dotenv()

tavily_client=TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)


