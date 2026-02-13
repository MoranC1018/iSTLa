from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional


@dataclass(frozen=True)
class OpenAICredentials:
    api_key: str
    organization: Optional[str] = None
    project: Optional[str] = None


def load_openai_credentials_from_csv(csv_path: str | Path = "user_api.csv") -> OpenAICredentials:
    """Load OpenAI API credentials from a one-row CSV file.

    Expected headers:
      - api_key (required)
      - openai_organization (optional)
      - openai_project (optional)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it from user_api.csv.example and put your API key in it."
        )

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty. It must contain one row with an api_key.")

    row = rows[0]
    api_key = (row.get("api_key") or "").strip()
    if not api_key:
        raise ValueError(f"{path} missing required column 'api_key' or it is blank.")

    org = (row.get("openai_organization") or "").strip() or None
    project = (row.get("openai_project") or "").strip() or None
    return OpenAICredentials(api_key=api_key, organization=org, project=project)
