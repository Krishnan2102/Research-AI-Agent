from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
import rag

mcp = FastMCP("RAG_Server", host="127.0.0.1", port=8001)


@mcp.tool()
def ingest_pdf(
    filepath: str,
    domain: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """
    Ingest a PDF file into the RAG vector store.

    The file must already exist on this server's local filesystem at
    `filepath`. Chunks are tagged with `domain`, which is later used by
    retrieve_from_rag / role_definition.py to restrict which roles can
    retrieve them. Files already ingested (matched by filepath) are
    skipped automatically.

    Args:
        filepath: Path to the PDF, readable by this server process.
        domain: Domain tag applied to every chunk from this file, e.g.
            "software_engineering". Should match one of the domains
            defined in role_definition.py.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.

    Returns:
        A dict with status ("success" | "skipped" | "error"), filepath,
        domain, and num_chunks (on success).
    """
    return rag.ingest_pdf(
        filepath=filepath,
        domain=domain,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@mcp.tool()
def retrieve_from_rag(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    allowed_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks from the local knowledge base for a
    query, without generating an LLM answer.

    Args:
        query: The search query text.
        top_k: Maximum number of chunks to return.
        score_threshold: Minimum cosine similarity (0-1) a chunk must
            have to be included.
        allowed_domains: If provided, restricts results to chunks tagged
            with one of these domains (see role_definition.py). Omit or
            pass null for unrestricted search across all domains.

    Returns:
        A list of chunks, each with id, content, metadata,
        similarity_score, distance, and rank.
    """
    return rag.retrieve_from_rag(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        allowed_domains=allowed_domains,
    )


if __name__ == "__main__":
    mcp.run(transport="sse")