import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from utils.logger import setup_logger
import urllib3

# File: loaders/web_loader.py
# Purpose: Download web pages and return their HTML content and title for preprocessing.

# Suppress SSL certificate warning logs to keep the console output cleaner.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = setup_logger("web_loader")

def load_url_content(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """
    Fetches the content of a URL and extracts clean textual content,
    along with page title.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        # Download the page and keep the HTML for later cleaning and chunking.
        logger.info(f"Fetching URL content: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        
        # Set encoding to apparent encoding so non-UTF-8 pages are read correctly.
        if response.encoding == 'ISO-8859-1' or not response.encoding:
            response.encoding = response.apparent_encoding or 'utf-8'
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        
        # Return raw soup and metadata
        return {
            "html": response.text,
            "title": title,
            "url": url
        }
    except requests.exceptions.RequestException as re_err:
        logger.error(f"HTTP request error fetching URL {url}: {str(re_err)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None

