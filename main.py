from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from tools import web_search, web_scrape

load_dotenv()

class Answer(BaseModel):
    topic: str
    key_points: str
    summary: str
    source: str

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

"""

tools = [web_search, web_scrape]

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools
)

structured_llm = model.with_structured_output(Answer)


def get_answer(query: str) -> dict:
    """Run one stateless turn: invoke the agent on the query, then
    synthesize the raw agent output into the structured Answer format."""
    raw_response = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )
    last_message = raw_response["messages"][-1].content

    final_answer = structured_llm.invoke(
        f"Synthesize the following research into the required format: {last_message}"
    )
    return final_answer.model_dump()


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
