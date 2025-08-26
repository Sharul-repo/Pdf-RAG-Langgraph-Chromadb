# 📄 PDF Agent with LangGraph, Chroma & Streamlit  

This project is an **AI-powered assistant** that allows you to upload PDFs, extract their contents, and interact with them using natural language.  
You can ask **questions** about the document, get **context-aware answers**, and even **summarize content** — all from a simple Streamlit UI.  

---

## 🚀 Features  

- 📂 Upload PDFs directly in the sidebar  
- 🔎 Automatic text extraction & chunking from PDFs  
- 💬 Ask natural language questions about the PDF  
- 🧠 Context-aware answers using vector similarity search  
- 🧹 Reset DB & clear uploaded files with one click  
- 🛠️ Uses **local embeddings** (no API cost!)  
- 💾 Maintains **conversation history** for context  

---

## 🛠️ Tech Stack  

- **[LangGraph](https://github.com/langchain-ai/langgraph)**  
  - Manages the agent workflow with nodes, edges, and conditional logic.  

- **[LangChain](https://www.langchain.com/)**  
  - Provides retrievers, embeddings, and agent orchestration.  

- **[Chroma](https://www.trychroma.com/)**  
  - Local vector database for storing and retrieving text embeddings.  

- **[HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**  
  - Embedding model (`all-MiniLM-L6-v2`) for semantic similarity.  

- **[Streamlit](https://streamlit.io/)**  
  - Interactive UI for file upload, chat, and DB reset.  

- **[PyPDF2](https://pypi.org/project/pypdf2/)**  
  - Extracts text from PDF files.  

- **[OpenAI-Compatible LLM (via LangChain ChatOpenAI wrapper)]**  
  - Powers the Q&A responses. Can be swapped with free or local models.  

---

## ⚙️ Installation & Setup  

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/pdf-qa-langgraph-chroma.git
cd pdf-qa-langgraph-chroma

# Install dependencies using uv (reads pyproject.toml + uv.lock)
uv sync

# Run your app
uv run streamlit run app.py
```