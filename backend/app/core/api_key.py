from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import os
import re
from typing import Optional


@dataclass(frozen=True)
class OpenAICredentials:
    api_key: str
    organization: Optional[str] = None
    project: Optional[str] = None


def _load_openai_credentials_from_env() -> OpenAICredentials | None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    organization = (os.getenv("OPENAI_ORGANIZATION") or "").strip() or None
    project = (os.getenv("OPENAI_PROJECT") or "").strip() or None
    return OpenAICredentials(api_key=api_key, organization=organization, project=project)


def load_openai_credentials_from_csv(csv_path: str | Path = "user_api.csv") -> OpenAICredentials:
    """Load OpenAI API credentials from a one-row CSV file.

    Expected headers:
      - api_key (required)
      - openai_organization (optional)
      - openai_project (optional)
    """
    env_creds = _load_openai_credentials_from_env()
    if env_creds is not None:
        return env_creds

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Set OPENAI_API_KEY or create it from user_api.csv.example and put your API key in it."
        )

    raw_text = path.read_text(encoding="utf-8")
    normalized_text = raw_text.replace("\\r\\n", "\n").replace("\\r", "\n")

    # Some users paste dashboard CSV snippets that include a literal "\\n"
    # instead of an actual newline. Recover that format when needed.
    if "\n" not in normalized_text and "\\n" in normalized_text:
        normalized_text = normalized_text.replace("\\n", "\n")

    reader = csv.DictReader(normalized_text.splitlines())
    rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty. It must contain one row with an api_key.")

    row = rows[0]
    api_key = (row.get("api_key") or "").strip()

    # Fallback for loose CSV headers/values (for example: "api_key, Personal, iSTLa\\n<key>").
    if not api_key:
        row_values = [str(v).strip() for v in row.values() if v is not None]
        key_pattern = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,})\b")
        for value in row_values:
            m = key_pattern.search(value)
            if m:
                api_key = m.group(1)
                break

    if not api_key:
        m = re.search(r"\b(sk-[A-Za-z0-9_-]{20,})\b", normalized_text)
        if m:
            api_key = m.group(1)

    if not api_key:
        raise ValueError(f"{path} missing required column 'api_key' or it is blank.")

    org = (row.get("openai_organization") or row.get("organization") or "").strip() or None
    project = (row.get("openai_project") or row.get("project") or "").strip() or None

    # Fallback for dashboard-style CSV where org/project are unnamed columns.
    if org is None or project is None:
        values = [str(v).strip() for v in row.values() if v is not None]
        non_key = [v for v in values if v and v != api_key]
        if org is None and non_key:
            org = non_key[0] or None
        if project is None and len(non_key) > 1:
            project = non_key[1] or None

    return OpenAICredentials(api_key=api_key, organization=org, project=project)
