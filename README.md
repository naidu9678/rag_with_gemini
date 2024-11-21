# Smart AI Chatbot

The **Smart AI Chatbot** is a Streamlit-based application that combines advanced document retrieval and language generation capabilities. It leverages LangChain's tools and Google Generative AI models to create a conversational AI experience optimized for question-answering tasks.

---

## Key Features

### 1. **PDF Document Loader**
- Uses the `PyPDFLoader` to load and process PDF documents for retrieval.

### 2. **Text Splitting**
- The `RecursiveCharacterTextSplitter` ensures efficient and meaningful text chunking for vectorization.

### 3. **Google Generative AI Embeddings**
- Embedding generation using Google's `embedding-001` model.

### 4. **Vector Store**
- Powered by FAISS, the chatbot uses a pre-trained vector store for fast and accurate similarity-based search.

### 5. **Language Model (LLM)**
- Powered by Google's `gemini-1.5-pro` model for high-quality natural language understanding and response generation.

### 6. **Retrieval-Augmented Generation (RAG)**
- Combines document retrieval with a question-answering chain to provide context-aware answers.

### 7. **Dynamic Prompting**
- Uses LangChain's `ChatPromptTemplate` for flexible and concise responses.

### 8. **User-Friendly Interface**
- Features interactive chat input and response history for seamless conversation.

---

## Usage

### Steps :
### Step1 : Clone the repository  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
### Step 2: Set Up Environment Variables

1. Create a `.env` file in the root of the project directory.
2. Add your Google API credentials in the following format:

### Step 3: Install Dependencies
Install all required Python libraries by running the following command:

pip install -r requirements.txt

### Step 4: Run the Application
Start the Streamlit chatbot application with this command:

streamlit run app.py

### Step 5: Interact with the Chatbot
 After starting the application, Streamlit will display a URL in the terminal (e.g., http://localhost:8501).
Open the URL in your web browser.

Enter your query in the chat interface, and the chatbot will:
Retrieve relevant context.
Provide concise and accurate responses.

