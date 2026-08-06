import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional
from embeddings.embedding_model import get_embedding_model
from utils.logger import setup_logger

# File: vectorstore/faiss_store.py
# Purpose: Save and load the local FAISS vector store used for retrieval.

logger = setup_logger("faiss_store")
# The FAISS index is stored in the faiss_index folder at the project root.
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")

def get_faiss_index(documents: Optional[List[Document]] = None, force_recreate: bool = False) -> Optional[FAISS]:
    """
    Loads or creates a FAISS index. If documents are provided and no index exists, 
    or force_recreate is True, a new index is created and saved.
    """
    embeddings = get_embedding_model()
    
    if force_recreate:
        if documents:
            logger.info("Creating a new FAISS index from documents (force_recreate).")
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(INDEX_DIR)
            return db
        else:
            logger.warning("Force recreate requested, but no documents provided. Deleting index.")
            clear_faiss_index()
            return None

    if os.path.exists(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        try:
            logger.info("Loading existing FAISS index from disk.")
            # Allow dangerous deserialization because it's locally saved and we control the contents
            db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            if documents:
                logger.info("Adding new documents to existing FAISS index.")
                db.add_documents(documents)
                db.save_local(INDEX_DIR)
            return db
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}. Recreating index.")
            if documents:
                db = FAISS.from_documents(documents, embeddings)
                db.save_local(INDEX_DIR)
                return db
            return None
    else:
        if documents:
            logger.info("Creating a new FAISS index from documents.")
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(INDEX_DIR)
            return db
        else:
            logger.info("No index found and no documents provided.")
            return None

def clear_faiss_index() -> None:
    """Deletes the locally stored FAISS index."""
    if os.path.exists(INDEX_DIR):
        logger.info(f"Removing FAISS index directory at {INDEX_DIR}")
        for file in os.listdir(INDEX_DIR):
            file_path = os.path.join(INDEX_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        try:
            os.rmdir(INDEX_DIR)
        except Exception as e:
            logger.error(f"Failed to remove directory {INDEX_DIR}: {e}")
