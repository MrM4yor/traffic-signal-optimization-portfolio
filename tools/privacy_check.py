"""Check tracked files and reachable Git history before portfolio publication.

This is a targeted path/credential audit, not a guarantee against every form of
private information. Findings report locations only, never matched secret values.
GitHub noreply commit emails and generic hosted-runner paths are allowed.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal_windows_path": re.compile(r"(?i)[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+"),
    "personal_unix_path": re.compile(r"/(?:Users|home)/[A-Za-z][^\s\"'/]+"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "api_secret": re.compile(r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b"),
    "private_email": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


def git(*arguments):
    """Read repository metadata without shell interpolation or external network."""
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True,
                          capture_output=True).stdout


def findings(text, location):
    """Report suspicious values by category/location without disclosing contents."""
    result = []
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if category == "private_email" and match.group().endswith("@users.noreply.github.com"):
                continue
            if category == "personal_unix_path" and match.group() == "/home/runner":
                continue
            result.append({"category": category, "location": location,
                           "line": text[:match.start()].count("\n") + 1})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    problems = []
    files = git("ls-files", "-z").decode().split("\0")
    for name in filter(None, files):
        problems.extend(findings((ROOT / name).read_text(encoding="utf-8", errors="replace"), name))
    blobs = set()
    commits = git("rev-list", "--all").decode().splitlines() if args.history else []
    for commit in commits:
        metadata = git("show", "-s", "--format=%an <%ae>%n%cn <%ce>%n%B", commit).decode()
        problems.extend(findings(metadata, "commit " + commit))
        for row in git("ls-tree", "-r", commit).decode().splitlines():
            header, path = row.split("\t", 1)
            oid = header.split()[2]
            if oid in blobs:
                continue
            blobs.add(oid)
            content = git("cat-file", "blob", oid).decode("utf-8", errors="replace")
            problems.extend(findings(content, commit + ":" + path))
    print(json.dumps({"tracked_files": len(list(filter(None, files))),
                      "history_commits": len(commits), "history_blobs": len(blobs),
                      "findings": problems}, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
