from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from tools import web_search,web_scrape

load_dotenv()

class Answer(BaseModel):
    topic:str
    key_points:str
    summary:str
    source:str

system_prompt="""

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

tools=[web_search,web_scrape]

model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
agent=create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools
)

question={"messages":[HumanMessage(content="What are the latest match results in Fifa?")]}



raw_response=agent.invoke(
    {"messages":[HumanMessage(content="Who is Tony Stark?")]}
)
last_message = raw_response["messages"][-1].content

structured_llm = model.with_structured_output(Answer)
final_answer = structured_llm.invoke(f"Synthesize the following research into the required format: {last_message}")

pprint(final_answer.model_dump())

