import subprocess
import time
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="smollm")

def start_ollama_server():
    # Start the Ollama server using subprocess
    print("Starting Ollama server...")
    return subprocess.Popen(["ollama", "serve"])
