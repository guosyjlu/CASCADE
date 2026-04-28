import os
import asyncio
from typing import List, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP


# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("rag")


# --------------------------------------------------------------------------- #
#  RAG Retrieve Function
# --------------------------------------------------------------------------- #

def _normalize_retrieve_url(url: str) -> str:
    """Ensure the URL points to the /retrieve endpoint."""
    if not url:
        return "http://localhost:9000/retrieve"
    if url.endswith("/retrieve"):
        return url
    if url.endswith("/"):
        return url + "retrieve"
    return url + "/retrieve"


def _get_retrieve_url(explicit_url: Optional[str] = None) -> str:
    """Resolve the RAG retrieval endpoint URL.

    Priority:
    1) explicit_url argument (base or full), normalized to /retrieve
    2) env RAG_RETRIEVE_URL (base or full), normalized to /retrieve
    3) default http://localhost:9000/retrieve
    """
    if explicit_url:
        return _normalize_retrieve_url(explicit_url)
    env_url = os.getenv("RAG_RETRIEVE_URL")
    if env_url:
        return _normalize_retrieve_url(env_url)
    return "http://localhost:9000/retrieve"


@mcp.tool()
async def retrieve(query: str) -> list[dict]:
    """
    Call the RAG retrieval service and return the most relevant documents.

    Parameters
    ----------
    query : str
        Single query string

    Returns
    -------
    list[dict]
        Retrieval results for the first query; if scores are enabled,
        elements are {document, score}
    """
    url = _get_retrieve_url()
    payload = {
        "queries": [query],
        # Force scores off by default
        "return_scores": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"RAG retrieve request failed: invalid JSON response: {getattr(resp, 'text', '')[:500]}")
    except requests.HTTPError as http_err:
        status = getattr(http_err.response, 'status_code', None)
        text = getattr(http_err.response, 'text', '')
        # Fallback: some servers require return_scores=True due to server-side unpacking
        if status and status >= 500 and payload.get("return_scores") is False:
            payload_fallback = dict(payload)
            payload_fallback["return_scores"] = True
            try:
                resp = requests.post(url, json=payload_fallback, timeout=30)
                resp.raise_for_status()
                try:
                    data = resp.json()
                except Exception:
                    raise RuntimeError(f"RAG retrieve request failed (fallback): invalid JSON response: {getattr(resp, 'text', '')[:500]}")
            except Exception as e:
                raise RuntimeError(f"RAG retrieve request failed (fallback): {e}; response={getattr(e, 'response', None) and getattr(e.response, 'text', '')}")
        else:
            raise RuntimeError(f"RAG retrieve request failed: {http_err}; response={text}")
    except Exception as e:
        raise RuntimeError(f"RAG retrieve request failed: {e}")

    # Expected shape: {"result": [[...]]}
    result = data.get("result")
    if not isinstance(result, list) or not result:
        return []
    first = result[0]
    if isinstance(first, list):
        return first
    return []


# --------------------------------------------------------------------------- #
#  MCPRAGCorpusTool Class (for direct Python usage)
# --------------------------------------------------------------------------- #

class MCPRAGCorpusTool:
    """RAG retrieval tool wrapper for synchronous/asynchronous usage in Python code."""

    def __init__(self, retrieve_url: Optional[str] = None):
        """Initialize RAG tool.

        Args:
            retrieve_url: Full /retrieve endpoint URL. If None, uses env RAG_RETRIEVE_URL
                          or defaults to http://localhost:9000/retrieve
        """
        self.retrieve_url = _get_retrieve_url(retrieve_url)

    def retrieve(self, query: str) -> List[Dict]:
        """Perform a synchronous retrieval for a single query.

        Args:
            query: Query string

        Returns:
            List of results for the query
        """
        payload = {
            "queries": [query],
            # Force scores off by default
            "return_scores": False,
        }

        try:
            resp = requests.post(self.retrieve_url, json=payload, timeout=30)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                raise RuntimeError(f"RAG retrieve request failed: invalid JSON response: {getattr(resp, 'text', '')[:500]}")
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, 'status_code', None)
            text = getattr(http_err.response, 'text', '')
            if status and status >= 500 and payload.get("return_scores") is False:
                payload_fallback = dict(payload)
                payload_fallback["return_scores"] = True
                try:
                    resp = requests.post(self.retrieve_url, json=payload_fallback, timeout=30)
                    resp.raise_for_status()
                    try:
                        data = resp.json()
                    except Exception:
                        raise RuntimeError(f"RAG retrieve request failed (fallback): invalid JSON response: {getattr(resp, 'text', '')[:500]}")
                except Exception as e:
                    raise RuntimeError(f"RAG retrieve request failed (fallback): {e}; response={getattr(e, 'response', None) and getattr(e.response, 'text', '')}")
            else:
                raise RuntimeError(f"RAG retrieve request failed: {http_err}; response={text}")
        except Exception as e:
            raise RuntimeError(f"RAG retrieve request failed: {e}")

        result = data.get("result")
        if not isinstance(result, list) or not result:
            return []
        first = result[0]
        return first if isinstance(first, list) else []

    async def retrieve_async(self, query: str) -> List[Dict]:
        """Perform an asynchronous retrieval for a single query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query)


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")

