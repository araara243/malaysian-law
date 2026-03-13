"""
AGC Malaysia Legal Acts Scraper

This module downloads Malaysian legal acts (PDFs) from the Attorney General's
Chambers (AGC) official website: https://lom.agc.gov.my/

Target acts for expanded RAG system (11 total):
- Contracts Act 1950 (Act 136)
- Specific Relief Act 1951 (Act 137)
- Housing Development (Control and Licensing) Act 1966 (Act 118)
- Sale of Goods Act 1957 (Act 382)
- Public Authorities (Control of Borrowing Powers) Act 1961 (Act 383)
- Partnership Act 1961 (Act 135)
- Penal Code (Act 574)
- Criminal Procedure Code (Act 593)
- National Land Code 1965 (Act 56)
- Strata Titles Act 1985 (Act 318)
- Courts of Judicature Act 1964 (Act 91)
"""

import os
import re
import time
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, cast
from urllib.parse import quote

# Add project root to sys.path for direct execution
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://lom.agc.gov.my"
PDF_BASE_EN = f"{BASE_URL}/ilims/upload/portal/akta/LOM/EN"
PDF_BASE_BM = f"{BASE_URL}/ilims/upload/portal/akta/LOM/BM"
ACT_DETAIL_URL = f"{BASE_URL}/act-detail.php"

# Expanded Acts to download (11 total)
EXPANDED_ACTS = [
    # === EXISTING ===
    {"act_no": 136, "name": "Contracts Act 1950"},
    {"act_no": 137, "name": "Specific Relief Act 1951"},
    {"act_no": 118, "name": "Housing Development (Control and Licensing) Act 1966"},
    # === COMMERCIAL ===
    {"act_no": 382, "name": "Sale of Goods Act 1957"},
    {"act_no": 383, "name": "Public Authorities (Control of Borrowing Powers) Act 1961"},
    {"act_no": 135, "name": "Partnership Act 1961"},
    # === CRIMINAL ===
    {"act_no": 574, "name": "Penal Code"},
    {"act_no": 593, "name": "Criminal Procedure Code"},
    # === PROPERTY ===
    {"act_no": 56, "name": "National Land Code"},
    {"act_no": 318, "name": "Strata Titles Act 1985"},
    # === CIVIL PROCEDURE ===
    {"act_no": 91, "name": "Courts of Judicature Act 1964"},
]

# Request headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Removes characters that are invalid in filenames on most operating systems.

    Args:
        name: The string to sanitize.

    Returns:
        A sanitized string safe for use in filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', '', name)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir() -> Path:
    """Get the raw data directory, creating it if necessary."""
    raw_dir = get_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_pdf(url: str, output_path: Path, retries: int = 3) -> bool:
    """
    Download a PDF file from a URL.
    
    Args:
        url: The URL of the PDF to download.
        output_path: The path to save the downloaded PDF.
        retries: Number of retry attempts on failure.
    
    Returns:
        True if download was successful, False otherwise.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Downloading: {url}")
            response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            
            if response.status_code == 200:
                # Verify it's actually a PDF
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                    logger.warning(f"Response is not a PDF: {content_type}")
                    return False
                
                # Write to file
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Saved: {output_path}")
                return True
            
            elif response.status_code == 404:
                logger.warning(f"PDF not found (404): {url}")
                return False
            
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed: "
                    f"HTTP {response.status_code}"
                )
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1}/{retries} error: {e}")
        
        # Wait before retry
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False


def validate_pdf_download(file_path: Path) -> bool:
    """
    Validate that a downloaded PDF is not corrupted or empty.

    Args:
        file_path: Path to the PDF file to validate.

    Returns:
        True if valid (exists, has content, within size limits), False otherwise.
    """
    if not file_path.exists():
        return False

    file_size = file_path.stat().st_size

    # Check file size limits (10KB minimum, 50MB maximum)
    MIN_SIZE = 10 * 1024  # 10KB
    MAX_SIZE = 50 * 1024 * 1024  # 50MB

    if file_size < MIN_SIZE:
        logger.warning(f"PDF too small: {file_path} ({file_size} bytes)")
        return False

    if file_size > MAX_SIZE:
        logger.warning(f"PDF too large: {file_path} ({file_size} bytes)")
        return False

    return True


def get_download_status(acts: list) -> dict:
    """
    Check download status for a list of Acts.

    Args:
        acts: List of dictionaries with 'act_no' and 'name' keys.

    Returns:
        Dictionary mapping act_no to status dict with 'exists', 'size', 'valid' keys.
    """
    output_dir = get_raw_data_dir()
    status = {}

    for act in acts:
        act_no = act["act_no"]
        act_name = act["name"]

        # Clean filename
        safe_name = sanitize_filename(act_name)
        filename = f"Act_{act_no}_{safe_name}_EN.pdf"
        file_path = output_dir / filename

        if file_path.exists():
            file_size = file_path.stat().st_size
            is_valid = validate_pdf_download(file_path)
            status[act_no] = {
                "exists": True,
                "size": file_size,
                "valid": is_valid,
                "path": file_path
            }
        else:
            status[act_no] = {
                "exists": False,
                "size": 0,
                "valid": False,
                "path": file_path
            }

    return status


def construct_pdf_url(act_no: int, language: str = "EN") -> str:
    """
    Construct the direct PDF URL for an Act.
    
    The AGC website uses a consistent URL pattern for the latest version:
    - English: /ilims/upload/portal/akta/LOM/EN/Act {act_no}.pdf
    - Malay: /ilims/upload/portal/akta/LOM/BM/Akta {act_no}.pdf
    
    Args:
        act_no: The Act number.
        language: "EN" for English, "BM" for Bahasa Malaysia.
    
    Returns:
        The constructed PDF URL.
    """
    if language.upper() == "EN":
        filename = f"Act {act_no}.pdf"
        base = PDF_BASE_EN
    else:
        filename = f"Akta {act_no}.pdf"
        base = PDF_BASE_BM
    
    # URL encode the filename (spaces become %20)
    encoded_filename = quote(filename)
    return f"{base}/{encoded_filename}"


def scrape_pdf_url_from_page(act_no: int, language: str = "BI") -> Optional[str]:
    """
    Scrape the PDF URL from the Act detail page.
    
    This is a fallback method if the direct URL pattern doesn't work.
    The page uses JavaScript to set the PDF source, so this method
    looks for the $src variable in the page scripts.
    
    Args:
        act_no: The Act number.
        language: "BI" for English Interface, "BM" for Malay.
    
    Returns:
        The PDF URL if found, None otherwise.
    """
    url = f"{ACT_DETAIL_URL}?language={language}&act={act_no}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        # Look for the $src variable in the page (older structure)
        match = re.search(r'\$src\s*=\s*["\']([^"\']+\.pdf)["\']', response.text)
        if match:
            pdf_url = match.group(1)
            logger.info(f"Found PDF URL in script: {pdf_url}")
            return pdf_url
        
        # New structure: Look for pdf.js viewer src or iframe src
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. Look for iframes (often used for pdf.js)
        for iframe in soup.find_all("iframe"):
            src = iframe.get("src", "")
            if "file=" in src:
                # Pattern: .../web/viewer.html?file=URL
                pdf_url = src.split("file=")[1].split("&")[0]
                # Unquote URL
                from urllib.parse import unquote
                pdf_url = unquote(pdf_url)
                if pdf_url.startswith("http"):
                    return pdf_url
                else:
                    return f"{BASE_URL}/{pdf_url.lstrip('/')}"
            
            if src.endswith(".pdf"):
                return src if src.startswith("http") else f"{BASE_URL}/{src.lstrip('/')}"
        
        # 2. Look for direct links with .pdf
        for link in soup.find_all("a", href=True):
            if link["href"].lower().endswith(".pdf"):
                pdf_url = link["href"]
                return pdf_url if pdf_url.startswith("http") else f"{BASE_URL}/{pdf_url.lstrip('/')}"

        logger.warning(f"Could not find PDF URL in page for Act {act_no}")
        return None
    
    except requests.RequestException as e:
        logger.error(f"Error scraping page for Act {act_no}: {e}")
        return None


def download_act(act_no: int, act_name: str, language: str = "EN") -> bool:
    """
    Download a specific Act PDF.
    
    Args:
        act_no: The Act number.
        act_name: The name of the Act (for filename).
        language: "EN" for English, "BM" for Bahasa Malaysia.
    
    Returns:
        True if download was successful, False otherwise.
    """
    output_dir = get_raw_data_dir()

    # Clean filename
    safe_name = sanitize_filename(act_name)
    filename = f"Act_{act_no}_{safe_name}_{language}.pdf"
    output_path = output_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        logger.info(f"Already exists: {output_path}")
        return True
    
    # Try direct URL first
    pdf_url = construct_pdf_url(act_no, language)
    if download_pdf(pdf_url, output_path):
        return True
    
    # Fallback: scrape URL from page
    logger.info(f"Direct URL failed, trying page scrape for Act {act_no}")
    scrape_lang = "BI" if language == "EN" else "BM"
    pdf_url = scrape_pdf_url_from_page(act_no, scrape_lang)
    
    if pdf_url:
        return download_pdf(pdf_url, output_path)
    
    logger.error(f"Failed to download Act {act_no}: {act_name}")
    return False


def download_expanded_acts() -> Dict[str, bool]:
    """
    Download all expanded Acts defined in the module.

    Returns:
        A dictionary with act numbers as keys and download status as values.
    """
    results: Dict[str, bool] = {}

    # Check current download status
    logger.info("Checking download status...")
    status = get_download_status(EXPANDED_ACTS)

    # Count existing valid downloads
    existing_count = sum(1 for s in status.values() if s["exists"] and s["valid"])
    missing_count = len(EXPANDED_ACTS) - existing_count

    logger.info("=" * 60)
    logger.info("Starting AGC Legal Acts Scraper - Expanded Mode")
    logger.info(f"Target directory: {get_raw_data_dir()}")
    logger.info(f"Total Acts: {len(EXPANDED_ACTS)}")
    logger.info(f"Already downloaded: {existing_count}")
    logger.info(f"Need to download: {missing_count}")
    logger.info("=" * 60)

    # Process only missing or invalid Acts
    download_queue = []
    for i, act in enumerate(EXPANDED_ACTS):
        act_no = act["act_no"]
        act_name = act["name"]

        if status[act_no]["exists"] and status[act_no]["valid"]:
            logger.info(f"[{i+1}/{len(EXPANDED_ACTS)}] Skipping: {act_name} (Act {act_no}) - Already downloaded")
            results[f"Act_{act_no}_EN"] = True
        else:
            if status[act_no]["exists"]:
                logger.info(f"[{i+1}/{len(EXPANDED_ACTS)}] Re-downloading: {act_name} (Act {act_no}) - Invalid file")
            else:
                logger.info(f"[{i+1}/{len(EXPANDED_ACTS)}] Queued: {act_name} (Act {act_no}) - Not found")
            download_queue.append((i, act))

    # Download missing Acts
    logger.info(f"\nStarting download of {len(download_queue)} Acts...\n")

    for i, act in download_queue:
        act_no = act["act_no"]
        act_name = act["name"]

        logger.info(f"[{i+1}/{len(download_queue)}] Processing: {act_name} (Act {act_no})")

        # Download English version
        en_success = download_act(act_no, act_name, "EN")

        # Validate the download
        if en_success:
            safe_name = sanitize_filename(act_name)
            filename = f"Act_{act_no}_{safe_name}_EN.pdf"
            file_path = get_raw_data_dir() / filename
            is_valid = validate_pdf_download(file_path)

            if not is_valid:
                logger.error(f"Downloaded file validation failed: {file_path}")
                en_success = False

        results[f"Act_{act_no}_EN"] = en_success

        # Small delay between requests to be polite
        if i < len(download_queue) - 1:
            time.sleep(1)

    # Detailed Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    logger.info("-" * 60)

    success_count: int = 0
    failed_count: int = 0

    for i, act in enumerate(EXPANDED_ACTS):
        act_no = act["act_no"]
        act_name = act["name"]
        success = results[f"Act_{act_no}_EN"]

        if success:
            status_text = "✓ SUCCESS"
            success_count = cast(int, success_count) + 1 # type: ignore
        else:
            status_text = "✗ FAILED"
            failed_count = cast(int, failed_count) + 1 # type: ignore

        logger.info(f"  [{cast(int, i)+1:2d}] Act {act_no:3d} - {act_name}: {status_text}") # type: ignore

        # Show file size if exists
        safe_name = sanitize_filename(act_name)
        filename = f"Act_{act_no}_{safe_name}_EN.pdf"
        file_path = get_raw_data_dir() / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"       File: {filename} ({size_mb:.2f} MB)")

    logger.info("-" * 60)
    logger.info(f"Total: {success_count} succeeded, {failed_count} failed")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    download_expanded_acts()
