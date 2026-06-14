from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.3,max_output_tokens=1000)

result=model.invoke('What is the capital of India?')

print(result.content)
