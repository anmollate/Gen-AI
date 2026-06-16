#Simple representation of using HuggingFaceEndpoint and ChatHuggingFace to answer the dynamic promts.
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct"
)

prompt=input("Enter Your Prompt:")

model=ChatHuggingFace(llm=llm)

result=model.invoke(prompt)

print(result.content)