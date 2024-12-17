import subprocess
import time
from langchain_ollama import OllamaLLM

def start_ollama_server():
    # Start the Ollama server
    print("Starting Ollama server...")
    return subprocess.Popen(["ollama", "serve"])

input_text = "Hello world"
result = start_ollama_server()

print(result)