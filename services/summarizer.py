import os
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from utils.logger import setup_logger
from utils.helpers import extract_text_content

logger = setup_logger("summarizer")

# File: services/summarizer.py
# Purpose: Create short summaries of the loaded knowledge base for the summary tab.

class SummaryResponse(BaseModel):
    executive: str = Field(description="A detailed professional paragraph summarizing the content.")
    bullets: str = Field(description="A list of 4-6 key details/factual points formatted as markdown bullet points.")
    takeaways: str = Field(description="3 actionable or major takeaway conclusions formatted as markdown bullet points.")

def get_summaries(db: Any, max_docs: int = 15, temperature: float = 0.3) -> Dict[str, str]:
    """
    Retrieves up to max_docs chunks from FAISS, combines them, and prompts the LLM
    to generate: Executive Summary, Bullet Summary, and Key Takeaways.
    Uses structured outputs with a regex-based fallback if the API fails structured response.
    """
    if not db:
        return {
            "executive": "No content loaded to summarize.",
            "bullets": "No content loaded to summarize.",
            "takeaways": "No content loaded to summarize."
        }
        
    try:
        # Retrieve all documents up to limit using docstore
        docstore = db.docstore._dict
        docs = list(docstore.values())[:max_docs]
        
        if not docs:
            return {
                "executive": "No content available.",
                "bullets": "No content available.",
                "takeaways": "No content available."
            }
            
        combined_text = ""
        for idx, doc in enumerate(docs):
            combined_text += f"Source {idx+1}: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}\n\n"
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        system_prompt = (
            "You are an expert summarizer. Analyze the text below and provide three separate summaries:\n"
            "1. Executive Summary: A detailed professional paragraph summarizing the content.\n"
            "2. Bullet Summary: A list of 4-6 key details/factual points.\n"
            "3. Key Takeaways: 3 actionable or major takeaway conclusions."
        )
        
        prompt = f"Text to summarize:\n\n{combined_text}"
        
        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(SummaryResponse)
            response = structured_llm.invoke(prompt)
            return {
                "executive": response.executive.strip(),
                "bullets": response.bullets.strip(),
                "takeaways": response.takeaways.strip()
            }
        except Exception as struc_e:
            logger.warning(f"Structured output failed, falling back to regex: {struc_e}")
            
            instructions = (
                f"{system_prompt}\n\n"
                "Please format your response precisely with the following markers, and nothing else:\n"
                "[EXECUTIVE]\n"
                "<executive summary text>\n"
                "[BULLETS]\n"
                "<bullet points>\n"
                "[TAKEAWAYS]\n"
                "<key takeaways>\n"
            )
            
            messages = [
                SystemMessage(content=instructions),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            resp_text = extract_text_content(response.content)
            
            executive = "Not generated"
            bullets = "Not generated"
            takeaways = "Not generated"
            
            # Use regex for parsing
            exec_match = re.search(r'(?:\[?EXECUTIVE\]?|##?\s*Executive Summary|##?\s*Executive)(.*?)(?:\[?BULLETS\]?|##?\s*Bullets|##?\s*Bullet Summary|\[?TAKEAWAYS\]?|##?\s*Takeaways|##?\s*Key Takeaways|$)', resp_text, re.DOTALL | re.IGNORECASE)
            bullet_match = re.search(r'(?:\[?BULLETS\]?|##?\s*Bullets|##?\s*Bullet Summary|##?\s*Key Points)(.*?)(?:\[?TAKEAWAYS\]?|##?\s*Takeaways|##?\s*Key Takeaways|$)', resp_text, re.DOTALL | re.IGNORECASE)
            takeaway_match = re.search(r'(?:\[?TAKEAWAYS\]?|##?\s*Takeaways|##?\s*Key Takeaways)(.*)', resp_text, re.DOTALL | re.IGNORECASE)
            
            if exec_match:
                executive = exec_match.group(1).strip()
            if bullet_match:
                bullets = bullet_match.group(1).strip()
            if takeaway_match:
                takeaways = takeaway_match.group(1).strip()
                
            if not exec_match and not bullet_match and not takeaway_match:
                executive = resp_text
                bullets = "See Executive Summary"
                takeaways = "See Executive Summary"
                
            return {
                "executive": executive,
                "bullets": bullets,
                "takeaways": takeaways
            }
            
    except Exception as e:
        logger.error(f"Error generating summaries: {str(e)}")
        return {
            "executive": f"Error generating summary: {str(e)}",
            "bullets": f"Error generating summary: {str(e)}",
            "takeaways": f"Error generating summary: {str(e)}"
        }
