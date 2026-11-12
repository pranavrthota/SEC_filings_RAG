"""Download real 10-K filings from EDGAR into ./filings.

    python -m evals.fetch_filings

SEC requires a descriptive User-Agent with a contact address on automated
requests, and rate-limits to 10 requests/second; both are honoured below.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

FILINGS_DIR = Path(__file__).resolve().parents[1] / "filings"

# ticker -> (CIK, how many of the most recent 10-Ks to take)
TARGETS = {
    "AAPL": ("0000320193", 2),
    "MSFT": ("0000789019", 1),
}

USER_AGENT = os.getenv("SEC_USER_AGENT", "sec-copilot-research contact@example.com")


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def fetch() -> list[Path]:
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for ticker, (cik, wanted) in TARGETS.items():
        submissions = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
        recent = submissions["filings"]["recent"]
        taken = 0
        for form, accession, document, filed in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
        ):
            if form != "10-K" or taken >= wanted:
                continue
            out = FILINGS_DIR / f"{ticker.lower()}-10k-{filed[:4]}.htm"
            if not out.exists():
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession.replace('-', '')}/{document}"
                )
                out.write_bytes(_get(url))
                time.sleep(0.2)  # stay well inside SEC's rate limit
            print(f"{ticker} FY{filed[:4]}: {out.name} ({out.stat().st_size / 1_000_000:.1f} MB)")
            written.append(out)
            taken += 1

    return written


if __name__ == "__main__":
    fetch()
