"""Web scraper utility to extract clean textual content from web pages."""

import logging
import re
from typing import Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def scrape_url_content(url: str, max_chars: int = 3500, timeout: int = 8) -> Optional[str]:
    """Extract readable text content from a given web URL.
    
    Args:
        url: The web URL to fetch and parse.
        max_chars: Maximum characters of text to return.
        timeout: Request timeout in seconds.
        
    Returns:
        Cleaned text string or None if fetching fails.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None

    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if response.status_code != 200:
            return None

        # Check content type
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove irrelevant tags
        for element in soup([
            "script", "style", "nav", "footer", "header", 
            "aside", "form", "svg", "noscript", "button", "iframe"
        ]):
            element.decompose()

        # Get text and clean whitespace
        text = soup.get_text(separator=" ")
        # Collapse multiple spaces and linebreaks
        cleaned_text = re.sub(r"\\s+", " ", text).strip()

        if not cleaned_text or len(cleaned_text) < 100:
            return None

        return cleaned_text[:max_chars]

    except Exception as e:
        logger.debug(f"Failed to scrape {url}: {e}")
        return None
