"""Utilities for maintaining LOG.md with reverse-chronological entries."""

import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "../../LOG.md")


def init_log():
    path = os.path.abspath(LOG_PATH)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# Experiment Log\n*Newest entries at top.*\n\n")


def log_entry(header: str, body: str):
    path = os.path.abspath(LOG_PATH)
    init_log()
    now = datetime.now().strftime("%H:%M")
    entry = f"## [{now}] {header}\n{body}\n\n"
    # Prepend after the title line
    with open(path, "r") as f:
        content = f.read()
    # Insert after second newline (after the subtitle)
    split_at = content.find("\n\n") + 2
    new_content = content[:split_at] + entry + content[split_at:]
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[LOG] {now} — {header}")
