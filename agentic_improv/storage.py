"""Atomic local checkpoints and portable exports."""

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .controller import Run


def json_export(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def run_document(run: Run) -> dict:
    return {"schema_version": 1, **asdict(run)}


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def save(self, run: Run) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{run.id}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False,
            ) as file:
                temporary = Path(file.name)
                file.write(json_export(run_document(run)))
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def archived_runs(self, current_id: str | None = None) -> list[dict]:
        runs = []
        for path in self.directory.glob("*.json"):
            if path.stem == current_id:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("schema_version") != 1 or data["id"] != path.stem:
                    continue
                if data["status"] in {"running", "paused"}:
                    data["status"] = "interrupted"
                    data["error"] = "The application closed before this experiment finished. The last saved replies are shown below."
                runs.append(data)
            except (OSError, ValueError, KeyError, AttributeError):
                continue
        return sorted(runs, key=lambda data: data.get("started_at", ""), reverse=True)


def csv_export(data: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "run_id", "status", "message_number", "agent", "speaker", "model", "timestamp_utc",
        "latency_ms", "input_tokens", "output_tokens", "message",
    ])
    writer.writerow([
        data["id"], data["status"], 0, "", "Initial message", "", data["started_at"],
        "", "", "", data["config"]["initial_message"],
    ])
    for turn in data["turns"]:
        reply = turn["reply"]
        writer.writerow([
            data["id"], data["status"], turn["number"], turn["agent"], turn["speaker"], reply["model"],
            turn["timestamp"], reply["latency_ms"], reply["input_tokens"], reply["output_tokens"], reply["text"],
        ])
    return output.getvalue()
