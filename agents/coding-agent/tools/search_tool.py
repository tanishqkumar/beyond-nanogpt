import os
from typing import List, Tuple, Optional, Dict, Any
import re 
import json  
from tools.base_tool import BaseTool
import requests
from bs4 import BeautifulSoup
from prompts.tool_prompts import SEARCH_TOOL_PROMPT
from pydantic import BaseModel 

# parses and cleans html from the url with BeautifulSoup and returns text on the page 
def get_page(url: str) -> str: 
    res = requests.get(url)
    html = res.text 
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text() + str(f'--'*10) + '\n'

def get_search_results(query: str, top_k: int = 5, deep: bool = False, max_len: int = 20_000) -> Tuple[List[str], bool]: 
    GOOGLE_SEARCH_KEY = os.environ.get('GOOGLE_SEARCH_KEY')
    SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')
    
    if not GOOGLE_SEARCH_KEY or not SEARCH_ENGINE_ID:
        raise Exception("Error: Missing required environment variables GOOGLE_SEARCH_KEY and/or SEARCH_ENGINE_ID")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_SEARCH_KEY, 
        'q': query, 
        'cx': SEARCH_ENGINE_ID,
    }

    res = requests.get(url, params=params)
    results = []
    total_len = 0

    if res.status_code == 200:
        data = res.json()
        items = data.get('items', [])
        
        if not deep:
            # Shallow search - return snippets
            for i, item in enumerate(items[:top_k], 1):  
                title = item.get('title', 'No title')
                link = item.get('link', 'No link')
                snippet = item.get('snippet', 'No snippet')
                result = f"Result {i}: {title}\nLink: {link}\n{snippet}"
                
                if total_len + len(result) > max_len:
                    break
                    
                results.append(result)
                total_len += len(result)
        else:
            # Deep search - scrape full page content
            for i, item in enumerate(items[:top_k], 1):  
                page_url = item.get('link', None)
                if page_url:
                    try:
                        page_content = get_page(page_url)
                        if total_len + len(page_content) > max_len:
                            # Truncate if needed
                            remaining_space = max_len - total_len
                            if remaining_space > 0:
                                results.append(page_content[:remaining_space])
                            break
                        results.append(page_content)
                        total_len += len(page_content)
                    except Exception as e:
                        print(f"Error scraping {page_url}: {e}")
                        continue
        
        print(f'Returning {len(results)} search results, total length: {total_len} chars')
        return results, True
    else:
        print(f"Error: {res.status_code}")
        print(res.text)
        return [], False

class SearchToolInput(BaseModel): 
    query: str 
    top_k: int = 3 # how many results we want 
    deep: bool = False # shallow or deep search 
    max_len: int = 20_000 # max num of characters we want in response we get back 

class SearchToolOutput(BaseModel): 
    query: str
    num_results: int
    results: List[str] = []

class SearchTool(BaseTool): 
    def __init__(
        self, 
        name: str = "search", 
        description: str = "Search the web for information", 
        prompt: str = SEARCH_TOOL_PROMPT,
        input_type: type = SearchToolInput,
    ):
        super().__init__(name, description, prompt, input_type)                

    def _execute(self, search_input: SearchToolInput) -> SearchToolOutput: 
        results, success = get_search_results(
            search_input.query, 
            search_input.top_k, 
            search_input.deep, 
            search_input.max_len, 
        )

        if not success: 
            print(f'WARNING: Failed Google Search API call in SearchTool with query: \n {search_input.query} \n')

        return SearchToolOutput(
            query=search_input.query, 
            num_results=len(results), 
            results=results, 
        ) 
