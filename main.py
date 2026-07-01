from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from tools import web_search, web_scrape
from rag import retrieve_from_rag

load_dotenv()

# Similarity threshold for deciding whether RAG's local knowledge is
# "good enough" to answer without falling back to web search/scrape.
# Tune this constant as needed.
RAG_SIMILARITY_THRESHOLD = 0.35
RAG_TOP_K = 5


class Answer(BaseModel):
    topic: str
    key_points: str
    summary: str
    source: str
    source_type: str  


system_prompt = """

"Role": "Research Agent"
"Aim": "Detailed Report of 100 words"
"Tools":
{"web_search": "search the web , return top 2 urls",
"web_scrape": scrape the urls, extract relevant information"}
"Result_format": 
"topic:topic name
key_points:numbered list of key points
summary:summary of the topic
source:numbered list of used urls"

Note: Local knowledge (RAG) is always checked first by the system before
you are invoked. You are only called when local knowledge was insufficient,
so proceed with web_search and web_scrape as normal.
"""

tools = [web_search, web_scrape]

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools
)

structured_llm = model.with_structured_output(Answer)


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


def _answer_from_web(query: str) -> dict:
    """Existing agent flow: web_search + web_scrape tools, then
    synthesize into the structured Answer format."""
    raw_response = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    last_message = raw_response["messages"][-1].content

    final_answer = structured_llm.invoke(
        f"Synthesize the following research into the required format: {last_message}"
    )
    result = final_answer.model_dump()
    result["source_type"] = "web"
    return result


def get_answer(query: str) -> dict:
    """
    RAG-first routing (enforced in code, not left to the LLM):

    1. Always try local RAG retrieval first.
    2. If retrieve_from_rag returns any chunks, they already meet
       RAG_SIMILARITY_THRESHOLD (the threshold is applied inside rag.py's
       retrieve_from_rag / RAGRetreiver.retrieve via score_threshold).
       In that case, answer ONLY from local knowledge — the agent and its
       web tools are never invoked.
    3. If RAG returns nothing (empty list — either no documents ingested,
       or nothing met the threshold), fall back to the existing agent flow
       which uses web_search and web_scrape.
    """
    rag_results = retrieve_from_rag(
        query,
        top_k=RAG_TOP_K,
        score_threshold=RAG_SIMILARITY_THRESHOLD,
    )

    if rag_results:
        return _answer_from_rag(query, rag_results)

    return _answer_from_web(query)


def main():
    print("Research Agent Chatbot")
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
            answer = get_answer(query)
            print()
            pprint(answer)
            print()
        except Exception as e:
            print(f"\n[Error] Something went wrong while processing that query: {e}\n")


if __name__ == "__main__":
    main()