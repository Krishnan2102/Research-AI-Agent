from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from typing import Dict,Any
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

load_dotenv()

tavily_client=TavilyClient()
user=UserAgent()

@tool
def web_search(query: str) -> str:

    """
       Search the web for information based on the query provided
       Then it returns a list of top 2 urls which the scraping tool can use
       to extract the required content
    """

    query_response=tavily_client.search(
        query,
        max_results=2
        )
    
    results = query_response.get('results', [])
    formatted_results = "\n".join([f"- {res['title']}: {res['url']}" for res in results])
    return formatted_results


@tool
def web_scrape(url:str) -> str:

    '''
    This the scraping tool you use after the web_search tool to scrape the list of 
    2 urls you get and find out all relevant information according to the query asked

    '''
    headers={"User-Agent":user.random}
    try:
        response=requests.get(url,headers=headers,timeout=10)
        soup=BeautifulSoup(response.text,'html.parser')

        for tag in soup(['script','style','nav','footer']):
            tag.decompose()

        text=soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        return clean_text
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"