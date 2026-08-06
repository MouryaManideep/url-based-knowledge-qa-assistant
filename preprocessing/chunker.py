from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document

# File: preprocessing/chunker.py
# Purpose: Split cleaned text into smaller chunks that can be stored in the vector database.

def chunk_web_content(sections: List[Dict[str, Any]], url: str, title: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Chunks sections of web content, preserving source metadata (URL, title, heading).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    documents = []
    for sec in sections:
        text = sec["text"]
        heading = sec["heading"]
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "title": title,
                    "heading": heading,
                    "type": "web",
                    "page": ""
                }
            )
            documents.append(doc)
            
    return documents

def chunk_pdf_content(pages_data: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Chunks pages of PDF content, preserving page numbers and source metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    documents = []
    for page in pages_data:
        text = page["text"]
        page_num = page["page_num"]
        source = page["source"]
        title = page["title"]
        
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "title": title,
                    "heading": f"Page {page_num}",
                    "type": "pdf",
                    "page": page_num
                }
            )
            documents.append(doc)
            
    return documents
