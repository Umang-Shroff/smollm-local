import subprocess
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import re
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
    """Load data from `data.txt` file and split it into chunks"""
    loader = TextLoader("data.txt")
    text = loader.load()[0].page_content

    # Use CharacterTextSplitter to break the content into chunks of 500 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embeddings = OllamaEmbeddings(model="smollm")
    persist_directory = "persist"
    
    # Create a Chroma vectorstore from the chunks
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    
    return vectorstore
    
def retrieve_context(query, vectorstore):
    """Retrieve relevant context from the data based on users' question."""
    # Perform similarity search to get the top k relevant documents
    docs = vectorstore.similarity_search(query, k=1)  # You can adjust k to get more context if necessary
    if docs:
        # Extract the most relevant document's content
        context = docs[0].page_content
        return context
    else:
        return "No relevant context found."
    

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
    print("\nWELCOME to SystemLLM! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Retrieve the relevant context for the current question
        relevant_context = retrieve_context(user_input, vectorstore)
        
        # Use the prompt with just the relevant context for the current question
        result = model.invoke({"question": user_input, "context": relevant_context})

        print("Bot: ", result)
    
    return result 
        
result = invoke_model()

print(result)