"""
Data loader for Week 6 Dashboard.
Loads real NVD CVE data from local JSON files (nvdcve-master dataset).
Parses CVSS v3.1 metrics, weaknesses, references, and configurations
into a flat DataFrame suitable for ML classification.
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd
import numpy as np

# === Paths ===
# Default path to the nvdcve-master dataset (configurable via env var)
NVD_DATA_DIR = Path(os.environ.get("NVD_DATA_DIR", r"C:\Users\Nitish\Downloads\nvdcve-master\nvdcve"))
CACHE_PATH = Path(__file__).parent / "cve_dataset.csv"


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        elif isinstance(current, list) and isinstance(k, int) and k < len(current):
            current = current[k]
        else:
            return default
        if current is None:
            return default
    return current


def _get_primary_cvss31(metrics: dict) -> Optional[dict]:
    """Extract the primary (NVD) CVSS v3.1 metric entry."""
    entries = metrics.get("cvssMetricV31", [])
    if not entries:
        return None
    # Prefer Primary (NVD) source
    for e in entries:
        if e.get("type") == "Primary":
            return e
    # Fall back to first entry
    return entries[0]


def _get_cvss2_score(metrics: dict) -> float:
    """Extract CVSS v2 base score."""
    entries = metrics.get("cvssMetricV2", [])
    if not entries:
        return 0.0
    for e in entries:
        if e.get("type") == "Primary":
            return float(_safe_get(e, "cvssData", "baseScore", default=0.0))
    return float(_safe_get(entries[0], "cvssData", "baseScore", default=0.0))


def _extract_weaknesses(weaknesses: list) -> List[str]:
    """Extract CWE IDs from weaknesses list."""
    cwes = []
    for w in (weaknesses or []):
        for desc in w.get("description", []):
            if desc.get("lang") == "en":
                val = desc.get("value", "")
                if val.startswith("CWE-") or val.startswith("NVD-"):
                    cwes.append(val)
    return cwes


def _count_cpe_vendors(configurations: list) -> int:
    """Count unique vendors from CPE configurations."""
    vendors = set()
    for config in (configurations or []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                # CPE format: cpe:2.3:a:vendor:product:...
                parts = criteria.split(":")
                if len(parts) >= 4:
                    vendors.add(parts[3])
    return len(vendors)


def _has_exploit_ref(references: list) -> int:
    """Check if any reference is tagged as an Exploit."""
    for ref in (references or []):
        tags = ref.get("tags", [])
        if "Exploit" in tags:
            return 1
    return 0


def parse_single_cve(filepath: str) -> Optional[dict]:
    """
    Parse a single NVD CVE JSON file into a flat feature dict.
    Returns None if the CVE lacks CVSS v3.1 data.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    cve_id = doc.get("id", "")
    metrics = doc.get("metrics", {})

    # Must have CVSS v3.1 for our classification target
    cvss31_entry = _get_primary_cvss31(metrics)
    if not cvss31_entry:
        return None

    cvss_data = cvss31_entry.get("cvssData", {})
    base_score = cvss_data.get("baseScore")
    if base_score is None or not isinstance(base_score, (int, float)):
        return None

    # CVSS v3.1 vector components
    av = cvss_data.get("attackVector", "UNKNOWN")
    ac = cvss_data.get("attackComplexity", "UNKNOWN")
    pr = cvss_data.get("privilegesRequired", "UNKNOWN")
    ui = cvss_data.get("userInteraction", "UNKNOWN")
    scope = cvss_data.get("scope", "UNKNOWN")
    conf = cvss_data.get("confidentialityImpact", "UNKNOWN")
    integ = cvss_data.get("integrityImpact", "UNKNOWN")
    avail = cvss_data.get("availabilityImpact", "UNKNOWN")

    # Exploitability and impact sub-scores
    exploit_score = cvss31_entry.get("exploitabilityScore", 0.0)
    impact_score = cvss31_entry.get("impactScore", 0.0)

    # CVSS v2
    cvss2_score = _get_cvss2_score(metrics)

    # Weaknesses
    weaknesses = _extract_weaknesses(doc.get("weaknesses", []))
    num_weaknesses = len(weaknesses)
    primary_cwe = weaknesses[0] if weaknesses else "NONE"

    # Configurations / vendors
    num_vendors = _count_cpe_vendors(doc.get("configurations", []))

    # References
    references = doc.get("references", [])
    num_references = len(references)
    has_exploit = _has_exploit_ref(references)

    # Temporal info
    published = doc.get("published", "")
    year_published = None
    if published:
        try:
            year_published = int(published[:4])
        except (ValueError, IndexError):
            pass

    # Vulnerability status
    vuln_status = doc.get("vulnStatus", "UNKNOWN")

    # Description length (proxy for complexity)
    desc_text = ""
    for d in doc.get("descriptions", []):
        if d.get("lang") == "en":
            desc_text = d.get("value", "")
            break
    desc_length = len(desc_text)

    # Severity label (target)
    if base_score >= 9.0:
        severity = "Critical"
    elif base_score >= 7.0:
        severity = "High"
    elif base_score >= 4.0:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "cve_id": cve_id,
        "cvss31_score": float(base_score),
        "cvss2_score": float(cvss2_score),
        "attack_vector": av,
        "attack_complexity": ac,
        "privileges_required": pr,
        "user_interaction": ui,
        "scope": scope,
        "confidentiality": conf,
        "integrity": integ,
        "availability": avail,
        "exploitability_score": float(exploit_score) if exploit_score else 0.0,
        "impact_score": float(impact_score) if impact_score else 0.0,
        "has_exploit": has_exploit,
        "num_vendors": num_vendors,
        "num_weaknesses": num_weaknesses,
        "primary_cwe": primary_cwe,
        "num_references": num_references,
        "desc_length": desc_length,
        "year_published": year_published,
        "vuln_status": vuln_status,
        "severity": severity,
    }


def load_from_nvd_files(
    data_dir: Optional[Path] = None,
    max_files: int = 0,
    year_min: int = 2015,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Load CVE data from local NVD JSON files.

    Args:
        data_dir: Path to directory containing CVE-*.json files.
        max_files: Maximum number of files to process (0 = all).
        year_min: Only include CVEs published from this year onward.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        DataFrame with one row per CVE that has CVSS v3.1 data.
    """
    data_dir = data_dir or NVD_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(f"NVD data directory not found: {data_dir}")

    # Get all JSON files
    pattern = str(data_dir / "CVE-*.json")
    all_files = glob.glob(pattern)

    if not all_files:
        raise FileNotFoundError(f"No CVE JSON files found in {data_dir}")

    # Filter by year if specified (quick filter on filename)
    if year_min:
        filtered = []
        for f in all_files:
            fname = os.path.basename(f)
            try:
                year = int(fname.split("-")[1])
                if year >= year_min:
                    filtered.append(f)
            except (IndexError, ValueError):
                filtered.append(f)
        all_files = filtered

    if max_files > 0:
        all_files = all_files[:max_files]

    total = len(all_files)
    print(f"Processing {total} CVE files from {data_dir}...")

    records = []
    skipped = 0
    for i, filepath in enumerate(all_files):
        if progress_callback and i % 5000 == 0:
            progress_callback(i, total)

        result = parse_single_cve(filepath)
        if result:
            records.append(result)
        else:
            skipped += 1

        if i % 20000 == 0 and i > 0:
            print(f"  Processed {i}/{total} files ({len(records)} valid, {skipped} skipped)")

    print(f"Done: {len(records)} CVEs with CVSS v3.1 out of {total} files ({skipped} skipped)")

    df = pd.DataFrame(records)
    return df


def save_cache(df: pd.DataFrame, path: Optional[Path] = None):
    """Save DataFrame to CSV cache."""
    path = path or CACHE_PATH
    df.to_csv(path, index=False)
    print(f"Cached {len(df)} records to {path}")


def load_cache(path: Optional[Path] = None) -> pd.DataFrame:
    """Load DataFrame from CSV cache."""
    path = path or CACHE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records from cache: {path}")
    return df


def load_data(
    use_cache: bool = True,
    max_files: int = 0,
    year_min: int = 2015,
    data_dir: Optional[Path] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Load CVE data. Tries CSV cache first, then parses NVD JSON files.

    Args:
        use_cache: If True, try loading from CSV cache first.
        max_files: Max JSON files to parse (0 = all).
        year_min: Minimum CVE year to include.
        data_dir: Override path to NVD data directory.
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        DataFrame with CVE features.
    """
    if use_cache and CACHE_PATH.exists():
        try:
            return load_cache()
        except Exception as e:
            print(f"Cache load failed: {e}, falling back to NVD files")

    df = load_from_nvd_files(
        data_dir=data_dir,
        max_files=max_files,
        year_min=year_min,
        progress_callback=progress_callback,
    )
    save_cache(df)
    return df


if __name__ == "__main__":
    print("=== NVD CVE Data Loader ===")
    df = load_data(use_cache=False)
    print(f"\nShape: {df.shape}")
    print(f"\nSeverity distribution:\n{df['severity'].value_counts()}")
    print(f"\nAttack Vector distribution:\n{df['attack_vector'].value_counts()}")
    print(f"\nYear range: {df['year_published'].min()} - {df['year_published'].max()}")
    print(f"\nSample:\n{df.head()}")
