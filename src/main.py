import asyncio
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from tools import web_search, web_scrape
from rag_mcp_client import call_rag_tool
from role_definition import get_allowed_domains, is_unrestricted

load_dotenv()

# Similarity threshold for deciding whether RAG's local knowledge is
# "good enough" to answer without falling back to web search/scrape.

RAG_SIMILARITY_THRESHOLD = 0.35
RAG_TOP_K = 5

class Answer(BaseModel):
    topic: str = Field(description="The main topic, title, or subject of the query. MUST be included.")
    key_points: str = Field(description="A detailed, numbered list of the most important key points.")
    summary: str = Field(description="A concise paragraph summarizing the answer.")
    source: str = Field(description="The document name, URLs, or source of the information.")
    source_type: str = Field(description="The type of source used (e.g., 'rag', 'web', 'pdf').")


class DomainClassification(BaseModel):
    domain: str = Field(
        description=(
            "The single best-matching domain for this query, chosen strictly "
            "from the candidate list provided. If the query does not clearly "
            "belong to any candidate domain, respond with exactly 'out_of_scope'."
        )
    )


OUT_OF_DOMAIN_MESSAGE = (
    "Sorry, I can't help you with that. This question is outside the "
    "domain(s) your role has access to."
)


def _rejection_answer(query: str) -> dict:
    """Structured Answer used when a restricted role asks an out-of-domain
    question. Short-circuits before RAG or the web agent are ever invoked."""
    return {
        "topic": query,
        "key_points": "N/A",
        "summary": OUT_OF_DOMAIN_MESSAGE,
        "source": "N/A",
        "source_type": "rejected",
    }


system_prompt = """

"Role": "Research Agent"
"Aim": "Detailed Report of 100 words"
"Tools":
{
  "web_search": "search the web, return top 2 urls",
  "web_scrape": "scrape the urls, extract relevant information",
  "search_wikipedia": "search Wikipedia for factual summaries and verified information on specific entities, concepts, or historical events."
}
"Tool_selection_rule":
"If the query asks for factual definitions, historical context, or general knowledge about a specific subject, try search_wikipedia first. If search_wikipedia returns no relevant results, or the topic is too niche/recent, fall back to web_search + web_scrape as normal."
"Result_format":
"topic:topic name
key_points:numbered list of key points
summary:summary of the topic
source:numbered list of used urls (or Wikipedia page names)"
Note: Local knowledge (RAG) is always checked first by the system before
you are invoked. You are only called when local knowledge was insufficient,
so proceed with the tools above as normal.
"""


async def _load_mcp_tools() -> list:
        client = MultiServerMCPClient(
            {
                "wikipedia_remote": {
                   
                    "url": "http://127.0.0.1:8000/sse", 
                    "transport": "sse",
                    
                    
                }
            }
        )
        return await client.get_tools()

 


model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


async def _build_agent(allowed_domains: Optional[List[str]] = None):
    """
    Build a fresh agent for each query, with a fresh MCP connection.

    Why not build this once at startup: the MCP client's connection is
    tied to the asyncio event loop that created it. asyncio.run() closes
    that loop as soon as tool-loading finishes, which kills the
    connection - so any *later* tool call fails with "Session
    terminated". Reconnecting per-query costs one extra round trip to
    the MCP server, but avoids using a dead connection.

    allowed_domains: if provided (restricted role), appended to the
    system prompt as a second layer of defense on top of the upstream
    domain classification gate in get_answer() -- web_search/web_scrape
    have no domain metadata to filter on, so this is prompt-level only,
    not enforced in code the way the RAG filter is.
    """

    mcp_tools = await _load_mcp_tools()
    tools = [web_search, web_scrape, *mcp_tools]

    agent_system_prompt = system_prompt
    if allowed_domains:
        agent_system_prompt += (
            f"\n\nSCOPE RESTRICTION: You may only research and answer "
            f"questions related to the following domain(s): "
            f"{', '.join(allowed_domains)}. If the user's question falls "
            f"outside these domains, do not attempt to answer it."
        )

    return create_agent(
        model=model,
        system_prompt=agent_system_prompt,
        tools=tools
    )

structured_llm = model.with_structured_output(Answer)
domain_classifier_llm = model.with_structured_output(DomainClassification)


def _classify_query_domain(query: str, candidate_domains: List[str]) -> str:
    """Ask the LLM which of the caller's allowed domains this query
    belongs to. Returns 'out_of_scope' if none fit. Only called for
    restricted roles -- unrestricted roles (admin) skip this entirely."""
    prompt = (
        "Classify which single domain the following research question "
        "belongs to.\n"
        f"Candidate domains: {candidate_domains}\n"
        "Respond with exactly one of the candidate domains, or "
        "'out_of_scope' if the question does not clearly belong to any "
        "of them.\n\n"
        f"Question: {query}"
    )
    result = domain_classifier_llm.invoke(prompt)
    return result.domain


def _answer_from_rag(query: str, rag_results: list) -> dict:
    """Build a structured Answer directly from RAG results, without
    invoking the agent or any web tools at all."""
    context_text = "\n\n---\n\n".join(
        f"[score={doc['similarity_score']:.3f} | file={doc.get('metadata', {}).get('source', 'unknown')}]\n{doc['content']}"
        for doc in rag_results
    )

    synthesis_prompt = (
        f"Using ONLY the following local knowledge base context, synthesize a research "
        f"answer in the required format for the query: '{query}'\n\n"
        f"CONTEXT:\n{context_text}"
    )

    final_answer = structured_llm.invoke(synthesis_prompt)
    result = final_answer.model_dump()
    result["source_type"] = "rag"
    return result


async def _answer_from_web(query: str, allowed_domains: Optional[List[str]] = None) -> dict:
    """Existing agent flow: web_search + web_scrape + arxiv_search tools,
    then synthesize into the structured Answer format."""
    agent = await _build_agent(allowed_domains=allowed_domains)
    raw_response = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]}
    )
    last_message = raw_response["messages"][-1].content

    final_answer = structured_llm.invoke(
        f"Synthesize the following research into the required format: {last_message}"
    )
    result = final_answer.model_dump()
    result["source_type"] = "web"
    return result


async def get_answer(query: str, role: str = "admin") -> dict:
    """
    Role-aware, RAG-first routing (enforced in code, not left to the LLM):

    0. If the role is restricted (not admin/unrestricted), classify the
       query's domain first. If it doesn't match one of the role's
       allowed domains, reject immediately -- neither RAG nor the web
       agent are ever invoked. Unrestricted roles skip this check.
    1. Otherwise, always try local RAG retrieval first, scoped to the
       role's allowed domains (allowed_domains=None for unrestricted
       roles means no filter is applied).
    2. If retrieve_from_rag returns any chunks, they already meet
       RAG_SIMILARITY_THRESHOLD (the threshold is applied inside rag.py's
       retrieve_from_rag / RAGRetreiver.retrieve via score_threshold).
       In that case, answer ONLY from local knowledge -- the agent and its
       web tools are never invoked.
    3. If RAG returns nothing (empty list -- either no documents ingested,
       nothing met the threshold, or nothing in-domain was found), fall
       back to the web agent, still scoped to allowed_domains via the
       system-prompt guardrail in _build_agent.
    """
    unrestricted = is_unrestricted(role)
    allowed_domains = get_allowed_domains(role)

    if not unrestricted:
        query_domain = _classify_query_domain(query, allowed_domains)
        if query_domain not in allowed_domains:
            return _rejection_answer(query)

    rag_results = await call_rag_tool(
        "retrieve_from_rag",
        query=query,
        top_k=RAG_TOP_K,
        score_threshold=RAG_SIMILARITY_THRESHOLD,
        allowed_domains=None if unrestricted else allowed_domains,
    )

    if rag_results:
        return _answer_from_rag(query, rag_results)

    return await _answer_from_web(
        query,
        allowed_domains=None if unrestricted else allowed_domains,
    )


def main():
    print("Research Agent Chatbot")
    role = input("Role (e.g. admin, software_engineer) [admin]: ").strip() or "admin"
    print(f"Running as role: {role}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        try:
            answer = asyncio.run(get_answer(query, role=role))
            print()
            pprint(answer)
            print()
        except Exception as e:
            import traceback
            print(f"\n[Error] Something went wrong while processing that query: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()