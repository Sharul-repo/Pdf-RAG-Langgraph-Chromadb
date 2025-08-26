import os
import shutil
import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict,List,Tuple,Optional
from langgraph.graph import StateGraph,START,END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

UPLOAD_FOLDER = "uploaded_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class DocState(TypedDict):
    uploaded_file: Optional[str]   
    vectorstore: Optional[Chroma]
    conversation_history: List[Tuple[str, str]]
    user_query: str
    answer: str


def safe_collection_dir(uploaded_file: str) -> str:
    """
    Create a unique Chroma collection directory for the uploaded file.
    If it already exists, reset it to ensure no contamination between PDFs.
    """
    file_stem = Path(uploaded_file).stem
    persist_dir = Path(".chroma_db") / file_stem

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return str(persist_dir)

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    extracted_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    return "\n".join(extracted_text).strip()

def reset_db():
    """
    Reset the vectorstore, conversation history, and uploaded file in session_state.
    Also deletes the Chroma database folder.
    """
    if os.path.exists(".chroma_db"):
        shutil.rmtree(".chroma_db")

    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)
    
    st.session_state.state = {
        "uploaded_file": None,
        "vectorstore": None,
        "conversation_history": [],
        "user_query": "",
        "answer": "",
    }

def document_extracter_agent(state:DocState)->DocState:
    uploaded_file=state.get("uploaded_file")
    if not uploaded_file:
        return {**state, "answer": "Please upload a PDF first."}
    
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    if not extracted_text:
        return {**state, "vectorstore": None, "answer": "Could not extract text from the PDF (it may be scanned images)."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(extracted_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = safe_collection_dir(uploaded_file)
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=str(persist_dir))

    return {
        **state,
        "vectorstore":vectorstore
    }

def query_agent(state: DocState)->DocState:
    vectorstore = state.get("vectorstore")
    user_query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    if not user_query:
        return {**state, "answer": "Ask me something about the document or anything else."}

    if vectorstore is None:
        return {**state, "answer": "No document uploaded yet."}

    # Retrieve docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""You are an assistant that answers questions.
If the user's question is related to the document, use the context below.
If it is not, answer normally.

Context:
{context}

Conversation History:
{history}

User Question:
{user_query}
"""

    # Use free LLM (swap with DeepSeek or any model you want)
    llm = ChatOpenAI(model="openai/gpt-oss-20b:free", temperature=0.3)
    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    # Update history
    updated_history = history + [(user_query, answer)]

    return {
        **state,
        "conversation_history": updated_history,
        "answer": answer
    }

def route_based_on_state(state: DocState) -> str:
    if state.get("vectorstore") is None:
        return "docextractagent"
    return "queryagent"

graph_builder=StateGraph(DocState)

graph_builder.add_node("docextractagent", document_extracter_agent)
graph_builder.add_node("queryagent", query_agent)

graph_builder.add_conditional_edges(
    START,
    route_based_on_state,
    {
        "docextractagent": "docextractagent",
        "queryagent": "queryagent"
    }
)

graph_builder.add_edge("docextractagent", "queryagent")

graph_builder.add_edge("queryagent", END)

graph = graph_builder.compile()

if "state" not in st.session_state:
    st.session_state.state = {
        "uploaded_file": None,
        "vectorstore": None,
        "conversation_history": [],
        "user_query": "",
        "answer": "",
    }

st.set_page_config(page_title="📄 PDF Q&A with LangGraph", layout="wide")
st.title("📄 PDF Q&A Agent")

with st.sidebar.expander("📂 Upload PDF", expanded=True):
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.state.update({
            "uploaded_file": file_path,
            "vectorstore": None,
            "conversation_history": [],
            "user_query": "",
            "answer": "",
        })
        st.success(f"Uploaded: {file_path}")
        if st.button("Reset Database", type="primary", use_container_width=True):
            reset_db()
            st.success("Database and history have been reset.")

if (st.session_state.state.get("conversation_history")):
    for utext, atext in st.session_state.state["conversation_history"]:
        if utext:
            st.chat_message("user").markdown(utext)
        if atext:
            st.chat_message("assistant").markdown(atext)


query = st.chat_input("💬 Ask a question about the document or anything else:")
if(query is not None and query!=""):
    with st.chat_message("Human"):
            st.markdown(query)
    st.session_state.state["user_query"] = query
    result =  graph.invoke(st.session_state.state) 
    st.session_state.state = result
    with st.chat_message("Human"):
            st.markdown(result["answer"])

# Reset button

