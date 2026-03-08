"""
Response Logger for tracking which model answered each query.

Useful for debugging and understanding output variations when switching
between different AI models.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LOG_FILE = Path("data/response_log.json")


def log_response(
    query: str,
    model_name: str,
    provider: str,
    answer: str,
    sources: List[Dict],
    retrieval_time_ms: float,
    generation_time_ms: float
) -> None:
    """
    Log a response with full metadata.

    Args:
        query: The user's question.
        model_name: Name of the AI model used.
        provider: LLM provider (openrouter, gemini).
        answer: The generated response.
        sources: List of retrieved source documents.
        retrieval_time_ms: Time taken to retrieve documents.
        generation_time_ms: Time taken to generate answer.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model": {
            "name": model_name,
            "provider": provider
        },
        "retrieval": {
            "time_ms": retrieval_time_ms,
            "source_count": len(sources)
        },
        "generation": {
            "time_ms": generation_time_ms,
            "total_time_ms": retrieval_time_ms + generation_time_ms
        },
        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
        "sources": [
            {
                "act_name": s.get("act_name", ""),
                "section_number": s.get("section_number", "")
            }
            for s in sources
        ]
    }

    # Load existing log
    logs = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

    logs.append(log_entry)

    # Save (keep last 1000 entries to prevent file bloat)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs[-1000:], f, indent=2, ensure_ascii=False)


def get_model_statistics(hours: int = 24) -> Dict:
    """
    Get statistics for specific models within a time window.

    Args:
        hours: Time window in hours (default 24).

    Returns:
        Dictionary with model names as keys and stats as values.
    """
    if not LOG_FILE.exists():
        return {}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(hours=hours)
    recent_logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) > cutoff]

    model_stats = {}
    for log in recent_logs:
        model = log["model"]["name"]
        if model not in model_stats:
            model_stats[model] = {"count": 0, "avg_time": [], "sources_per_query": []}

        model_stats[model]["count"] += 1
        model_stats[model]["avg_time"].append(log["generation"]["total_time_ms"])
        model_stats[model]["sources_per_query"].append(log["retrieval"]["source_count"])

    # Calculate averages
    for model in model_stats:
        if model_stats[model]["avg_time"]:
            model_stats[model]["avg_time_ms"] = sum(model_stats[model]["avg_time"]) / len(model_stats[model]["avg_time"])
        if model_stats[model]["sources_per_query"]:
            model_stats[model]["avg_sources"] = sum(model_stats[model]["sources_per_query"]) / len(model_stats[model]["sources_per_query"])

    return model_stats


def get_recent_logs(limit: int = 50) -> List[Dict]:
    """Get the most recent response logs.

    Args:
        limit: Number of recent logs to return.

    Returns:
        List of log entries sorted by timestamp (newest first).
    """
    if not LOG_FILE.exists():
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    # Return last N entries, reversed for newest first
    return logs[-limit:][::-1]


def compare_models(hours: int = 24) -> str:
    """Generate a comparison report of model performance.

    Args:
        hours: Time window for comparison.

    Returns:
        Formatted string comparing model statistics.
    """
    stats = get_model_statistics(hours)

    if not stats:
        return "No logs available for comparison."

    report = ["\nModel Performance Comparison (Last {} hours)\n".format(hours)]
    report.append("=" * 60)

    for model, data in sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True):
        report.append(f"\n{model}:")
        report.append(f"  Queries: {data['count']}")
        report.append(f"  Avg Response Time: {data.get('avg_time_ms', 0):.1f}ms")
        report.append(f"  Avg Sources/Query: {data.get('avg_sources', 0):.1f}")

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def export_logs(output_path: Optional[Path] = None) -> None:
    """Export logs to a file.

    Args:
        output_path: Optional path to save logs. If None, saves to desktop.
    """
    if not LOG_FILE.exists():
        print("No logs to export.")
        return

    if output_path is None:
        from pathlib import Path
        output_path = Path.home() / "Desktop" / f"mylaw_rag_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    import shutil
    shutil.copy2(LOG_FILE, output_path)
    print(f"Logs exported to: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "stats":
            hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
            print(compare_models(hours))
        elif command == "recent":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            recent = get_recent_logs(limit)
            for i, log in enumerate(recent, 1):
                print(f"\n[{i}] {log['timestamp']}")
                print(f"Model: {log['model']['name']} ({log['model']['provider']})")
                print(f"Query: {log['query']}")
                print(f"Time: {log['generation']['total_time_ms']:.1f}ms")
                print(f"Preview: {log['answer_preview']}")
                print("-" * 40)
        elif command == "export":
            output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            export_logs(output_path)
        else:
            print("Available commands:")
            print("  python response_logger.py stats [hours]  - Show model statistics")
            print("  python response_logger.py recent [limit]  - Show recent logs")
            print("  python response_logger.py export [path]  - Export logs to file")
    else:
        # Default: show recent logs
        recent = get_recent_logs(10)
        for i, log in enumerate(recent, 1):
            print(f"\n[{i}] {log['timestamp']}")
            print(f"Model: {log['model']['name']} ({log['model']['provider']})")
            print(f"Query: {log['query']}")
            print(f"Time: {log['generation']['total_time_ms']:.1f}ms")
            print(f"Preview: {log['answer_preview']}")
            print("-" * 40)
