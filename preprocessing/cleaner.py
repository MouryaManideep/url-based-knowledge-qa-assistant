from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List

# File: preprocessing/cleaner.py
# Purpose: Remove noisy HTML parts like menus, footers, scripts, and ads before chunking.

def clean_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Cleans raw HTML by stripping ads, scripts, navbars, sidebars, styles.
    Returns list of sections/paragraphs with their associated nearest headings.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove HTML sections that usually do not contain the main document content.
    unwanted_tags = [
        "script", "style", "nav", "footer", "header", "aside", 
        "noscript", "iframe", "form", "button", "svg"
    ]
    for tag in soup(unwanted_tags):
        tag.decompose()
        
    # Also find and remove classes/ids that commonly represent ads/sidebars
    for tag in soup.find_all(attrs={"class": re.compile(r"ad-|sidebar|footer|menu|navbar|nav|cookie", re.I)}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": re.compile(r"ad-|sidebar|footer|menu|navbar|nav", re.I)}):
        tag.decompose()

    sections = []
    current_heading = "Introduction"
    
    # Traverse the page body and keep heading-based structure for easier retrieval.
    body = soup.body if soup.body else soup
    
    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]):
        name = element.name
        text = element.get_text().strip()
        if not text:
            continue
            
        if name.startswith("h"):
            current_heading = text
        elif name == "table":
            # Format tables as pipe-separated content for better LLM retrieval reading
            rows = []
            for row in element.find_all("tr"):
                cells = [cell.get_text().strip() for cell in row.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                table_text = "\n".join(rows)
                sections.append({
                    "text": table_text,
                    "heading": current_heading
                })
        elif name == "li":
            sections.append({
                "text": f"- {text}",
                "heading": current_heading
            })
        else:
            sections.append({
                "text": text,
                "heading": current_heading
            })
            
    # If sections is empty but there is text, fallback to getting all paragraph texts
    if not sections:
        text = soup.get_text()
        # Basic cleanup of multiple newlines
        text = re.sub(r'\n+', '\n', text)
        sections.append({
            "text": text,
            "heading": "General Content"
        })
        
    return sections

