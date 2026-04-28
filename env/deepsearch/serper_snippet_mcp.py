import json
import requests
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("serperapi")

# --------------------------------------------------------------------------- #
#  Tools
# --------------------------------------------------------------------------- #

@mcp.tool()
async def google_search(query: str) -> list[str]:
    """
    Run a Google search and only return the snippet results.
    
    Parameters
    ----------
    query : str
        The search query string (e.g., "Coffee")
    
    Returns
    -------
    list[dict]
        The list of snippet results from Google.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
        })
    headers = {
        'X-API-KEY': '<YOUR_SERPER_API_KEY>',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()
    
    organic = results.get("organic", [])
    
    tidy_results = []
    for o in organic:
        title = o.get("title", "")
        snippet = o.get("snippet", "")
        tidy = title + "\n" + snippet
        tidy_results.append(tidy)
        
    return tidy_results

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")