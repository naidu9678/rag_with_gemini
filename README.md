Smart AI Chatbot
The Smart AI Chatbot is a Streamlit-based application that combines advanced document retrieval and language generation capabilities. It leverages LangChain's tools and Google Generative AI models to create a conversational AI experience optimized for question-answering tasks.

Key Features
PDF Document Loader

Use the PyPDFLoader to load and process PDF documents for retrieval.
Text Splitting

The RecursiveCharacterTextSplitter ensures efficient and meaningful text chunking for vectorization.
Google Generative AI Embeddings

Embedding generation using Google's embedding-001 model.
Vector Store

Powered by FAISS, the chatbot uses a pre-trained vector store for fast and accurate similarity-based search.
Language Model (LLM)

Powered by Google's gemini-1.5-pro model for high-quality natural language understanding and response generation.
Retrieval-Augmented Generation (RAG)

Combines document retrieval with a question-answering chain to provide context-aware answers.
Dynamic Prompting

Uses LangChain's ChatPromptTemplate for flexible and concise responses.
User-Friendly Interface

Interactive chat input and response history for seamless conversation.
