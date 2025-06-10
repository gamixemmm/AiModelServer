from duckduckgo_search import DDGS
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """Clean the search result text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text.strip()

def search_sap_info(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search for SAP-related information using DuckDuckGo.
    Returns a list of dictionaries containing titles and snippets.
    """
    try:
        with DDGS() as ddgs:
            # Add 'SAP' to the query if it's not already there
            if 'sap' not in query.lower():
                query = f"SAP {query}"
            
            # Perform the search
            results = []
            for r in ddgs.text(query, max_results=max_results):
                if r:
                    results.append({
                        'title': clean_text(r['title']),
                        'snippet': clean_text(r['body'])
                    })
            
            return results
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results into a readable string."""
    if not results:
        return "No search results found."
    
    formatted = "Here's what I found from recent web searches:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. {result['title']}\n"
        formatted += f"   {result['snippet']}\n\n"
    
    return formatted 