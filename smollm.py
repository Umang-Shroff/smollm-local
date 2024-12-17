import subprocess
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings


import os


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
    
def load_data():
    """Load data from `data.txt` file"""
    loader = TextLoader("data.txt")
    embeddings = OllamaEmbeddings(model="smollm")
    persist_directory = "persist"
    vectorstore = Chroma.from_documents(loader.load(), embeddings, persist_directory=persist_directory)
    return vectorstore
    
def retrieve_context(query, vectorstore):
    """Retrieve relevant context from the data based on users question."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    result = retriever.get_relevant_documents(query)
    return result[0].page_content if result else ""
    
def invoke_model():
    ollama_process = start_ollama_server()
    
    # Allow some time for the server to initialize
    time.sleep(5)
    
    vectorstore = load_data()
    
    model = OllamaLLM(model="smollm")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    final_context=handle_conversation(chain, vectorstore)
    # result = chain.invoke({"context":final_context, "question":""})       
    # (input=input_text)
    
    stop_ollama_server(ollama_process)
    # return result

def handle_conversation(model, vectorstore):
    context=""
    print("\nWECOME to SystemLLM! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        relevant_context = retrieve_context(user_input, vectorstore)
        
        context += f"\nUser: {user_input}\nAI: {relevant_context}"

        result = model.invoke({"context":context, "question":user_input})
        print("Bot: ", result)
        context+= f"\nUser: {user_input}\nAI: {result}"
    return context
        
result = invoke_model()

print(result)