import subprocess
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


template = """
Answer the question below

Here is the conversation history: {context}

Question: {question}

Answer: 
""" 


def start_ollama_server():
    # Start the Ollama server
    print("Initiating Ollama server...")
    return subprocess.Popen(["ollama", "serve"])

def stop_ollama_server(process):
    # Stop the Ollama server
    print("Terminating Ollama server...")
    process.terminate()
    process.wait()
    
def invoke_model(input_text):
    ollama_process = start_ollama_server()
    
    # Allow some time for the server to initialize
    time.sleep(5)
    
    model = OllamaLLM(model="smollm")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = chain.invoke({"context":"", "question":input_text})       
    # (input=input_text)
    
    stop_ollama_server(ollama_process)
    return result

input_text = "Hello world"
result = invoke_model(input_text)

print(result)