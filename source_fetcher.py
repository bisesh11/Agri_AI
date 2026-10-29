"""Source Fetcher Module.

Safely fetches permitted web page content, parses main article text using
BeautifulSoup, removes navigational/advertising noise, respects timeouts,
and extracts available publication metadata without violating access policies.
"""

import logging
import re
from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("source_fetcher")

USER_AGENT = "AgriChatBot/1.0 (Educational Academic Project; Agricultural Extension Research; +https://github.com/agricultural-ai)"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
}


def is_valid_http_url(url: str) -> bool:
    """Validates that a URL is a legitimate HTTP/HTTPS web address."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc or "." not in parsed.netloc:
            return False
        # Block internal/private network addresses
        if any(parsed.netloc.startswith(p) for p in ("localhost", "127.", "192.168.", "10.", "172.16.")):
            return False
        return True
    except Exception:
        return False


def extract_metadata_date(soup: BeautifulSoup) -> Optional[str]:
    """Attempts to discover article publication or update date from HTML meta tags."""
    date_meta_names = [
        "article:published_time",
        "article:modified_time",
        "datePublished",
        "dateModified",
        "DC.date.issued",
        "DC.date.created",
        "publication_date",
        "og:updated_time"
    ]
    for name in date_meta_names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            content = tag["content"].strip()
            # Clean ISO timestamps into YYYY-MM-DD
            match = re.match(r"^(\d{4}-\d{2}-\d{2})", content)
            if match:
                return match.group(1)
            return content[:25]
    return None


def fetch_page_content(
    url: str,
    timeout: int = 5,
    max_chars: int = 4000
) -> Optional[Dict[str, str]]:
    """Retrieves permitted webpage content, extracts clean article text, and parses metadata.

    Args:
        url: Web URL to fetch.
        timeout: Maximum seconds to wait.
        max_chars: Character limit to prevent excessive memory usage.

    Returns:
        Dict with keys: 'url', 'title', 'text', 'published_date', or None on failure.
    """
    if not is_valid_http_url(url):
        logger.warning(f"Invalid URL rejected: {url}")
        return None

    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            stream=False
        )

        if response.status_code != 200:
            logger.info(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None

        # Verify content type is HTML
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            logger.info(f"Skipping non-HTML content type for {url}: {content_type}")
            return None

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Strip scripts, styles, forms, and navigation chrome
        for unwanted in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe", "svg"]):
            unwanted.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text().strip()

        # Extract publication date
        published_date = extract_metadata_date(soup)

        # Target main content container if available
        main_container = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|post|body", re.I))
        target_elem = main_container if main_container else soup.body or soup

        # Extract clean paragraphs
        paragraphs = [p.get_text().strip() for p in target_elem.find_all(["p", "li", "h2", "h3"])]
        meaningful_paras = [p for p in paragraphs if len(p.split()) >= 6]

        if not meaningful_paras:
            # Fallback to general text extraction
            raw_text = target_elem.get_text(separator=" ", strip=True)
            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
        else:
            cleaned_text = " ".join(meaningful_paras)

        if not cleaned_text:
            return None

        # Limit text length
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars] + "..."

        return {
            "url": url,
            "title": title or url,
            "text": cleaned_text,
            "published_date": published_date or "Not specified",
        }

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout occurred while fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting content from {url}: {e}")
        return None

