import subprocess
import time
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
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
    """Load data from `data.txt` file"""
    loader = TextLoader("data.txt")
    embeddings = OllamaEmbeddings(model="smollm")
    persist_directory = "persist"
    vectorstore = Chroma.from_documents(loader.load(), embeddings, persist_directory=persist_directory)
    
    # query = "sample query"  # You can adjust this to test any query you like
    # docs = vectorstore.similarity_search(query, k=5)
    # for doc in docs:
    #     print(doc)
    
    return vectorstore
    
def retrieve_context(query, vectorstore):
    """Retrieve relevant context from the data based on users question."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    result = retriever.similarity_search(query, k=1)
    if result:
        full_content = result[0].page_content
        # Perform a manual string search to find matches to the query
        matched_content = perform_manual_search(query, full_content)
        return matched_content
    return ""
    
def perform_manual_search(query, content):
    """Manually search within the content for matching portions based on the query."""
    # Use regex to find occurrences of the query term in the content (case-insensitive)
    matched_data = re.findall(r'.{0,30}' + re.escape(query) + r'.{0,30}', content, flags=re.IGNORECASE)
    # Join and return the matched segments (with some context around the query)
    return ' '.join(matched_data) if matched_data else "No relevant information found."

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