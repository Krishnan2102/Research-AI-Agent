from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage
from pprint import pprint
from tools import web_search

load_dotenv()

class Answer(BaseModel):
    main_content:str
    tools_used:str

system_prompt="""
You are a research agent. When the question is asked you have to generate a deatiled response of 100 words

If the user asks about a person tell these things in pointers about them:
Name:
Born:
Died:
About:

"""

tools=[web_search]

model=ChatGroq(model="llama-3.3-70b-versatile")
agent=create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=tools,
    response_format=Answer
)

question={"messages":[HumanMessage(content="What are the latest match results in Fifa?")]}



response=agent.invoke(
    {"messages":[HumanMessage(content="Who is better Messi?")]}
)
"""
for token in agent.stream(
    {"messages": [HumanMessage(content="What are the latest match results in Fifa World Cup?")]},
    stream_mode="messages"
)


    
    
    if token.content:  
        print(token.content, end="", flush=True)  
"""

pprint(response['structured_response'])
