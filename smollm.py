import subprocess
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


template = """
Answer the question below with only what is asked and do not provide any other extra text aside from the question asked

Here is the conversation history: {context}

Question: {question}

Answer: 
""" 


def start_ollama_server():
    # Start the Ollama server
    print("\nInitiating Ollama server...")
    return subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,   
        stderr=subprocess.PIPE    
    )

def stop_ollama_server(process):
    # Stop the Ollama server
    print("\nTerminating Ollama server...")
    process.terminate()
    process.wait()
    
def invoke_model():
    ollama_process = start_ollama_server()
    
    # Allow some time for the server to initialize
    time.sleep(5)
    
    model = OllamaLLM(model="smollm")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    final_context=handle_conversation(chain)
    # result = chain.invoke({"context":final_context, "question":""})       
    # (input=input_text)
    
    stop_ollama_server(ollama_process)
    # return result

def handle_conversation(model):
    context=""
    print("\nWECOME to SystemLLM! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = model.invoke({"context":context, "question":user_input})
        print("Bot: ", result)
        context+= f"\nUser: {user_input}\nAI: {result}"
    return context
        
result = invoke_model()

print(result)