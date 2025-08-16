This is a **Retrieval-Augmented Generation (RAG)** powered chatbot that allows you to upload any PDF document and ask natural language questions about it. It retrieves relevant document chunks using vector embeddings and responds using Cohere's `command-r` LLM model.

---

##  Features

- Upload any PDF and extract contextual answers
- Uses **Cohere** for embedding + generation (`command-r`)
- Stores vector embeddings in **Pinecone** for fast semantic retrieval
- Reranks results using `rerank-english-v2.0` for improved accuracy
- Streamlit interface with real-time chat


```bash

Project Structure
RAG_QA_Bot/
├── frontend.py         # Streamlit UI application
├── chatbot.py          # Chat interface & LLM response logic
├── embedding_utils.py  # PDF chunking, embedding with Cohere, and retrieval with Pinecone
├── requirements.txt    # Python dependencies (Streamlit, Cohere, Pinecone, PyMuPDF)
└── README.md           # Project documentation
```

## 🔧 Setup Instructions

git clone https://github.com/theepankumargandhi/rag-document-qa-bot.git
cd rag-document-qa-bot

conda create -n rag_env python=3.10 -y
conda activate rag_env

pip install -r requirements.txt

streamlit run frontend.py

---

🔑 API Keys Required
This project uses the following services:

Cohere API Key: For embedding, reranking and LLM generation

Pinecone API Key: For storing and querying document embeddings

You can get keys from:

https://dashboard.cohere.com

https://app.pinecone.io

These keys are securely entered through the Streamlit sidebar.

---

How It Works (RAG Flow)

Upload PDF → text extracted via PyMuPDF

Text is split into chunks

Chunks are embedded using Cohere embed-english-v3.0

Stored in Pinecone vector DB

Ask question → vector search → top-k chunks retrieved

Results are reranked using rerank-english-v2.0

Final answer is generated via command-r using the top reranked chunks

---

Example Use Cases


Internal company document Q&A

PDF policy summarization

Academic paper assistants

Legal contract QA



