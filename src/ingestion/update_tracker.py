"""
Document Update Tracker for MyLaw-RAG

Tracks when documents were last updated and triggers
re-ingestion if changes are detected at the AGC website.
"""

import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# Path to metadata storage
METADATA_FILE = Path("data/update_metadata.json")

# AGC Act URLs for checking updates
ACT_URLS = {
    "Contracts Act 1950": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20181026",
    "Specific Relief Act 1951": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20181027",
    "Partnership Act 1961": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20220804",
    "Sale of Goods Act 1957": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20220805",
    "Housing Development Act 1966": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20220801",
    "Strata Titles Act 1985": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20220803",
    "Courts of Judicature Act 1964": "https://lom.agc.gov.my/act-view.php?id=UA%20Prs%20220802"
}


def calculate_pdf_hash(pdf_path: Path) -> str:
    """
    Calculate SHA-256 hash of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Hex string of the SHA-256 hash.
    """
    if not pdf_path.exists():
        return ""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def get_agc_last_modified(act_url: str) -> Optional[str]:
    """
    Get the last modified date from AGC website headers.

    Args:
        act_url: URL to download from AGC.

    Returns:
        Last-Modified header value or None if not available.
    """
    try:
        response = requests.head(act_url, timeout=10)
        last_modified = response.headers.get("Last-Modified")
        return last_modified
    except Exception as e:
        print(f"Warning: Could not check AGC for {act_url}: {e}")
        return None


def load_metadata() -> Dict:
    """Load or create metadata file.

    Returns:
        Dictionary containing all metadata.
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: Dict) -> None:
    """Save metadata to file.

    Args:
        metadata: Dictionary to save.
    """
    # Ensure data directory exists
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def check_needs_update(act_name: str, act_url: str, current_pdf_hash: str) -> bool:
    """
    Check if document needs re-ingestion.

    Args:
        act_name: Name of the Act (e.g., "Contracts Act 1950").
        act_url: URL to download from AGC.
        current_pdf_hash: Hash of currently stored PDF.

    Returns:
        True if re-ingestion is needed, False otherwise.
    """
    metadata = load_metadata()

    # Get stored hash for this act
    stored_hash = metadata.get("documents", {}).get(act_name, {}).get("pdf_hash")

    # If hashes differ, update needed
    if stored_hash != current_pdf_hash:
        print(f"  {act_name}: Hash changed (re-ingestion needed)")
        print(f"    Stored: {stored_hash[:16]}...")
        print(f"    Current: {current_pdf_hash[:16]}...")
        return True

    # Check AGC website for changes
    agc_modified = get_agc_last_modified(act_url)
    if agc_modified:
        stored_modified = metadata.get("documents", {}).get(act_name, {}).get("agc_last_modified")
        if agc_modified != stored_modified:
            print(f"  {act_name}: AGC website updated (re-ingestion needed)")
            print(f"    Stored: {stored_modified}")
            print(f"    Current: {agc_modified}")
            return True

    return False


def update_metadata_after_ingestion(act_name: str, act_url: str, pdf_hash: str, chunk_count: int) -> None:
    """
    Update metadata after successful ingestion.

    Args:
        act_name: Name of the Act.
        act_url: URL to download from AGC.
        pdf_hash: Hash of the ingested PDF.
        chunk_count: Number of chunks created.
    """
    metadata = load_metadata()

    if "documents" not in metadata:
        metadata["documents"] = {}

    metadata["documents"][act_name] = {
        "pdf_hash": pdf_hash,
        "agc_last_modified": get_agc_last_modified(act_url),
        "chunk_count": chunk_count,
        "last_ingested": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

    metadata["last_check"] = datetime.now().isoformat()
    save_metadata(metadata)

    print(f"  {act_name}: Metadata updated ({chunk_count} chunks)")


def check_all_documents(data_dir: Path = Path("data/raw")) -> List[str]:
    """
    Check all documents for updates.

    Args:
        data_dir: Directory containing PDF files.

    Returns:
        List of act names that need re-ingestion.
    """
    acts_to_update = []

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return acts_to_update

    for pdf_file in data_dir.glob("*.pdf"):
        # Extract act name from filename
        act_name = pdf_file.stem.replace("_", " ")

        # Get corresponding AGC URL
        act_url = ACT_URLS.get(act_name)

        if not act_url:
            print(f"  Warning: No AGC URL for {act_name}, skipping...")
            continue

        # Calculate current hash
        current_hash = calculate_pdf_hash(pdf_file)

        # Check if update needed
        if check_needs_update(act_name, act_url, current_hash):
            acts_to_update.append(act_name)

    return acts_to_update


def get_status() -> Dict:
    """
    Get status of all documents.

    Returns:
        Dictionary with document status information.
    """
    metadata = load_metadata()
    return metadata


def print_status_report() -> None:
    """Print a formatted status report of all documents."""
    metadata = get_status()

    print("\n" + "=" * 70)
    print("Document Update Status Report")
    print("=" * 70)

    if "documents" not in metadata:
        print("\n  No documents tracked yet.")
        return

    print(f"\nLast checked: {metadata.get('last_check', 'Never')}")
    print(f"\nTracked Documents ({len(metadata.get('documents', {}))}):\n")

    for act_name, doc_info in metadata.get("documents", {}).items():
        last_ingested = doc_info.get("last_ingested", "Unknown")
        agc_modified = doc_info.get("agc_last_modified", "Unknown")
        chunk_count = doc_info.get("chunk_count", 0)

        # Calculate age
        try:
            last_date = datetime.fromisoformat(last_ingested)
            age = (datetime.now() - last_date).days
            age_str = f"{age} days ago"
        except:
            age_str = "Unknown"

        print(f"  {act_name}:")
        print(f"    Chunks: {chunk_count}")
        print(f"    Last ingested: {last_ingested} ({age_str})")
        print(f"    AGC modified: {agc_modified}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            print("Checking for document updates...")
            acts_to_update = check_all_documents()
            if acts_to_update:
                print(f"\nUpdates needed for: {', '.join(acts_to_update)}")
                print("\nRun full ingestion pipeline to update:")
                print("  1. python src/ingestion/agc_scraper.py")
                print("  2. python src/ingestion/text_extractor.py")
                print("  3. python src/ingestion/chunker.py")
                print("  4. python src/ingestion/vector_ingest.py")
            else:
                print("\nNo updates found. All documents are up to date.")

        elif command == "status":
            print_status_report()

        else:
            print("Available commands:")
            print("  python update_tracker.py check       - Check for document updates")
            print("  python update_tracker.py status      - Show current document status")
    else:
        # Default: check and show status
        acts_to_update = check_all_documents()
        print_status_report()

        if acts_to_update:
            print(f"\nUpdates needed for: {', '.join(acts_to_update)}")
            print("\nRun full ingestion pipeline to update:")
            print("  1. python src/ingestion/agc_scraper.py")
            print("  2. python src/ingestion/text_extractor.py")
            print("  3. python src/ingestion/chunker.py")
            print("  4. python src/ingestion/vector_ingest.py")
