from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import time

start_time = time.time()


llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.3, #temperature controls the deterministic or random nature of the output. Lower values make the output more deterministic, while higher values make it more random.
        "max_new_tokens": 1000
    }
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of America?")
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
print(result.content)

