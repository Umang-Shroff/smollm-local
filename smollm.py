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

input_text = "Hello world"
result = start_ollama_server()

print(result)