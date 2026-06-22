from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage
from pprint import pprint
from tools import web_search,web_scrape

load_dotenv()

class Answer(BaseModel):
    topic:str
    key_points:str
    summary:str
    url:str

system_prompt="""

You are a research agent. When the question is asked you have to generate a deatiled response of 100 words
using the following tools at your disposal provided in your tools attribute:

1.web_search: you search the web based on the provided query and return a list of top 2 urls
2.web_scrape: based on the urls you get, you scrape the page for relvant content and then return the desired result according to the provided format

topic:name of the topic
key_points:key point regardind the topic in a numbered list format
summary:summary and conclusion of the topic
url:a numbererd list of urls you used to get the response

"""

tools=[web_search,web_scrape]

model=ChatGroq(model="llama-3.3-70b-versatile")
agent=create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools,
    response_format=Answer
)

question={"messages":[HumanMessage(content="What are the latest match results in Fifa?")]}



raw_response=agent.invoke(
    {"messages":[HumanMessage(content="What is Project Insight in MCU?")]}
)
"""
for token in agent.stream(
    {"messages": [HumanMessage(content="What are the latest match results in Fifa World Cup?")]},
    stream_mode="messages"
)


    
    
    if token.content:  
        print(token.content, end="", flush=True)  
"""

structured_llm = model.with_structured_output(Answer)
final_answer = structured_llm.invoke(f"Synthesize the following research into the required format: {raw_response}")

pprint(final_answer.model_dump())
#pprint(response['structured_response'])
