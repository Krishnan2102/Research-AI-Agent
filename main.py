from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

class Answer(BaseModel):
    main_content:str

system_prompt="""
You are a research agent. When the question is asked you have to generate a deatiled response of 100 words

"""

model=ChatGroq(model="llama-3.3-70b-versatile")
agent=create_agent(
    model=model,
    system_prompt=system_prompt,
    response_format=Answer
)

question=HumanMessage(content="How did World War 1 End?")

response=agent.invoke(
    {"messages":[question]}
)

pprint(response['structured_response'])
