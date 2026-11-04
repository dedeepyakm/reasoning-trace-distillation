"""Logging helpers: results, errors, and visible command runs.

Three things, all writing to the `logs/` folder at the repo root:
  log_result(...)  -> logs/RESULT_LOG.csv   one row per number, with its source file and script
  log_error(...)   -> logs/ERROR_LOG.md     one table row per error, with cause and fix
  run_step(...)    -> runs a command, shows its output live, saves it to logs/runs/,
                      and writes an ERROR_LOG row automatically if it fails
"""
import csv
import datetime
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
RUN_DIR = os.path.join(LOG_DIR, "runs")
RESULT_CSV = os.path.join(LOG_DIR, "RESULT_LOG.csv")
ERROR_MD = os.path.join(LOG_DIR, "ERROR_LOG.md")
RESULT_FIELDS = ["timestamp", "label", "metric", "value", "source_file", "script", "git_commit", "notes"]


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log_result(label, metric, value, source_file, script, notes=""):
    """Append one number to logs/RESULT_LOG.csv. Every number needs its source file and script."""
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(RESULT_CSV)
    if not new:
        key = (str(label), str(metric), str(value), str(source_file))
        with open(RESULT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["label"], r["metric"], r["value"], r["source_file"]) == key:
                    return
    with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": _now(), "label": label, "metric": metric, "value": value,
                    "source_file": source_file, "script": script, "git_commit": _git_commit(),
                    "notes": notes})


def log_error(where, error, cause="unknown", fix="not fixed yet", status="open"):
    """Append one row to the table at the end of logs/ERROR_LOG.md."""
    os.makedirs(LOG_DIR, exist_ok=True)

    def clean(s):
        return str(s).replace("|", "/").replace("\n", " ").strip()

    row = "| {} | {} | {} | {} | {} | {} |\n".format(
        _now(), clean(where), clean(error)[:300], clean(cause), clean(fix), clean(status))
    with open(ERROR_MD, "a", encoding="utf-8") as f:
        f.write(row)


def run_step(name, cmd, cwd=None):
    """Run a command, print its output as it happens, save it, log failures. Returns the exit code."""
    os.makedirs(RUN_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RUN_DIR, "{}_{}.txt".format(stamp, name))
    print("=== {} ===\n$ {}\n".format(name, " ".join(cmd)))
    tail = []
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("$ {}\n\n".format(" ".join(cmd)))
        p = subprocess.Popen(cmd, cwd=cwd or REPO_ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            tail.append(line)
            tail = tail[-15:]
        p.wait()
        f.write("\n[exit code {}]\n".format(p.returncode))
    print("\n[exit code {}; output saved to {}]".format(p.returncode, os.path.relpath(out_path, REPO_ROOT)))
    if p.returncode != 0:
        last = " ".join(l.strip() for l in tail if l.strip())[-300:]
        log_error(where=name, error=last, cause="see " + os.path.relpath(out_path, REPO_ROOT),
                  fix="not fixed yet", status="open")
        print("Error logged to logs/ERROR_LOG.md (fill in cause and fix by hand once known).")
    return p.returncode
