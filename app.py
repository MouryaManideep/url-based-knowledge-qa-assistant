import os
# Disable Streamlit file watcher for smoother local development in this project.
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# File: app.py
# Purpose: Main entry point for the Streamlit web app.
# Flow: user provides URLs/PDFs -> content is processed -> embeddings are stored -> chat and summaries use the knowledge base.

# Load environment variables such as the Gemini API key from a local .env file.
load_dotenv()

# Setup page configuration for the web interface.
st.set_page_config(
    page_title="URL-Based RAG QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Dark glassmorphic headers and backgrounds */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #4D96FF 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    .citation-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f4f8;
        border-left: 4px solid #4D96FF;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    
    .dark .citation-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

from loaders.web_loader import load_url_content
from loaders.pdf_loader import load_pdf_content
from preprocessing.cleaner import clean_html
from preprocessing.chunker import chunk_web_content, chunk_pdf_content
from vectorstore.faiss_store import get_faiss_index, clear_faiss_index
from rag.qa_chain import generate_answer
from services.summarizer import get_summaries
from utils.helpers import is_valid_url, export_chat_to_markdown

# Initialize session state so information stays available across Streamlit reruns.
# This helps the app remember chat history, processed sources, and the FAISS database.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = set()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "db" not in st.session_state:
    # Try loading an existing FAISS index from disk when the app starts.
    st.session_state.db = get_faiss_index()

# Main page title shown at the top of the app.
st.title("URL-Based Intelligent QA Assistant")

# Sidebar Configuration for user inputs and settings.
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="If you don't have it set in your .env file, enter it here."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    # Strict Source Mode Checkbox
    strict_mode = st.checkbox(
        "Strict Source Mode",
        value=True,
        help="When checked, the assistant only answers using the provided sources. When unchecked, it will fall back to general knowledge if the answer isn't in the sources."
    )
        
    st.markdown("---")
    
    # Dynamic settings for RAG and LLM
    st.subheader("🤖 Model & RAG Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Lower value is more deterministic, higher is more creative.")
    search_type = st.selectbox("Search Strategy", options=["similarity", "mmr"], index=0, format_func=lambda x: "Similarity Search" if x == "similarity" else "Max Marginal Relevance (MMR)")
    top_k = st.slider("Retrieval Chunk Count (K)", min_value=1, max_value=15, value=5, step=1)
    
    st.subheader("✂️ Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=150, step=50)

    st.markdown("---")
    
    # Input URLs
    st.subheader("🌐 Web Sources")
    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com\nhttps://another-page.com"
    )
    
    st.markdown("---")
    
    # Upload PDFs
    st.subheader("📄 PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Build Knowledge Base Action
    build_db = st.button("Build Knowledge Base . . . . ")
    
    # Clear index action
    if st.button("Clear Database . . . . "):
        clear_faiss_index()
        st.session_state.db = None
        st.session_state.processed_urls.clear()
        st.session_state.processed_files.clear()
        st.session_state.chat_history.clear()
        st.success("Database cleared!")
        st.rerun()

# Process the knowledge base when the user clicks the build button.
if build_db:
    # Split the multiline URL text into a clean list of links.
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
    invalid_urls = [u for u in urls if not is_valid_url(u)]
    valid_urls = [u for u in urls if is_valid_url(u)]
    
    if invalid_urls:
        st.sidebar.warning(f"Skipped invalid URLs: {', '.join(invalid_urls)}")
        
    all_documents = []
    
    # Process Web URLs
    if valid_urls:
        with st.spinner("Processing Web URLs..."):
            for url in valid_urls:
                if url in st.session_state.processed_urls:
                    continue
                content = load_url_content(url)
                if content:
                    sections = clean_html(content["html"])
                    chunks = chunk_web_content(sections, url=url, title=content["title"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    all_documents.extend(chunks)
                    st.session_state.processed_urls.add(url)
                    
    # Process Uploaded PDFs
    if uploaded_files:
        with st.spinner("Processing PDF Documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in st.session_state.processed_files:
                    continue
                # Load PDF from bytes
                file_bytes = uploaded_file.read()
                pages_data = load_pdf_content(file_bytes, uploaded_file.name)
                chunks = chunk_pdf_content(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_documents.extend(chunks)
                st.session_state.processed_files.add(uploaded_file.name)
                
    if all_documents:
        with st.spinner("Generating embeddings & updating FAISS index..."):
            db = get_faiss_index(all_documents)
            st.session_state.db = db
        st.sidebar.success(f"Successfully processed {len(all_documents)} new chunks!")
        st.rerun()
    else:
        st.sidebar.info("No new documents or URLs were added.")

# Tabs for Main View: Chat, Summarization, Document Sources
tab_chat, tab_summary, tab_sources = st.tabs(["💬 Chat Assistant", "📝 Summarization", "📚 Active Sources"])

with tab_sources:
    st.subheader("Currently Loaded Sources")
    
    if st.session_state.db:
        try:
            chunk_count = len(st.session_state.db.docstore._dict)
            st.info(f"📊 **Database Stats**: Currently storing **{chunk_count} chunks** in local FAISS vector store.")
        except Exception:
            st.info("📊 **Database Stats**: Chunks successfully stored in vector database.")
    else:
        st.info("📊 **Database Stats**: Vector database is currently empty.")
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Webpages:**")
        if st.session_state.processed_urls:
            for url in st.session_state.processed_urls:
                st.markdown(f"- [{url}]({url})")
        else:
            st.info("No webpages loaded.")
    with col2:
        st.markdown("**PDF Documents:**")
        if st.session_state.processed_files:
            for filename in st.session_state.processed_files:
                st.markdown(f"- 📄 {filename}")
        else:
            st.info("No PDF files uploaded.")

with tab_summary:
    st.subheader("Source Material Summary")
    if not st.session_state.db:
        st.warning("Please build the knowledge base first to generate summaries.")
    else:
        if st.button("⚡ Generate Summaries"):
            with st.spinner("Analyzing sources and generating summaries..."):
                summaries = get_summaries(st.session_state.db, temperature=temperature)
                st.markdown("### 📊 Executive Summary")
                st.write(summaries["executive"])
                st.markdown("---")
                st.markdown("### 📌 Key Points")
                st.write(summaries["bullets"])
                st.markdown("---")
                st.markdown("### 💡 Takeaways")
                st.write(summaries["takeaways"])

with tab_chat:
    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("🔍 Show Source Citations"):
                    for cite in message["citations"]:
                        title = cite.get("title", "N/A")
                        heading = cite.get("heading", "N/A")
                        page = cite.get("page", "")
                        page_str = f", Page {page}" if page else ""
                        source = cite.get("source", "N/A")
                        st.markdown(
                            f"<div class='citation-card'>"
                            f"<strong>{title}</strong> (Section: {heading}{page_str})<br>"
                            f"Source: <a href='{source}' target='_blank'>{source}</a>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Chat input
    if question := st.chat_input("Ask a question about the provided sources..."):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Please enter a Gemini API Key in the sidebar or check your .env configuration.")
        elif not st.session_state.db:
            st.warning("Please populate the knowledge base first by adding URLs or PDFs.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Searching knowledge base and thinking..."):
                    answer, retrieved_docs = generate_answer(
                        question=question,
                        db=st.session_state.db,
                        chat_history=st.session_state.chat_history[:-1], # Exclude current question
                        strict=strict_mode,
                        k=top_k,
                        temperature=temperature,
                        search_type=search_type
                    )
                    
                    message_placeholder.markdown(answer)
                    
                    # Extract unique citations
                    citations = []
                    seen_citations = set()
                    for doc in retrieved_docs:
                        key = (doc.metadata.get("title"), doc.metadata.get("heading"), doc.metadata.get("page"))
                        if key not in seen_citations:
                            seen_citations.add(key)
                            citations.append({
                                    "title": doc.metadata.get("title"),
                                    "heading": doc.metadata.get("heading"),
                                    "page": doc.metadata.get("page"),
                                    "source": doc.metadata.get("source")
                            })
                            
                    # Show citations below answer
                    if citations:
                        with st.expander("🔍 Show Source Citations"):
                            for cite in citations:
                                title = cite["title"]
                                heading = cite["heading"]
                                page = cite["page"]
                                page_str = f", Page {page}" if page else ""
                                source = cite["source"]
                                st.markdown(
                                    f"<div class='citation-card'>"
                                    f"<strong>{title}</strong> (Section: {heading}{page_str})<br>"
                                    f"Source: <a href='{source}' target='_blank'>{source}</a>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                
            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })

    # Export & download options
    if st.session_state.chat_history:
        st.markdown("---")
        md_export = export_chat_to_markdown(st.session_state.chat_history)
        st.download_button(
            label="📥 Export Chat History to Markdown",
            data=md_export,
            file_name="chat_history.md",
            mime="text/markdown"
        )

