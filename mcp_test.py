

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    model="openai/gpt-oss-120b",  
    input="What models are trending on Hugging Face right now?",
    tools=[
        {
            "type": "mcp",
            "server_label": "huggingface",
            "server_url": "https://huggingface.co/mcp",
            
        }
    ],
)

print(response.output_text)