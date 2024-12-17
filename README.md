# Smollm-local 

## Overview

Smollm-local is a conversational AI system that uses a locally loaded LLM (model=smollm) through the **Ollama** platform to retrieve relevant information from a custom dataset stored in a `data.txt` file. The system allows for text retrieval from this file based on user queries and incorporates a conversation chain to respond to user inputs.

## Features

- **Text Retrieval**: The system reads data from a `data.txt` file and retrieves relevant information based on user queries.
- **Ollama LLM**: The system uses the locally loaded Ollama LLM model for processing queries.
- **Conversation Chain**: The system supports a conversational interface, where the context of previous user inputs is maintained throughout the session.
- **Query-based Context Filtering**: The retrieved context is filtered and presented to the user in response to their questions.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
- `pip` (Python package installer)
- **Ollama**: Ensure you have Ollama installed and running locally. You can download Ollama from [here](https://ollama.com/).

### Required Python Packages

The project relies on several Python libraries. You can install them using `pip`. Run the following command in your terminal:

```bash
pip install langchain langchain-ollama langchain-core langchain-community
