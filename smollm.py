import subprocess
import time
from langchain_ollama import OllamaLLM

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
    result = model.invoke(input=input_text)
    
    stop_ollama_server(ollama_process)
    return result

input_text = "Hello world"
result = invoke_model(input_text)

print(result)