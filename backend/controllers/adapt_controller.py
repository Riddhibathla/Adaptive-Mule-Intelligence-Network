from __future__ import annotations

import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from config import (
    ADAPT_API_BASE_URL,
    ADAPT_API_KEY,
    ADAPT_DOWNLOAD_PATH,
    ADAPT_INSTRUCTION_PATH,
    DATASET_PATH,
)
class AdaptApiError(RuntimeError):
    pass


class AdaptClient:
    def __init__(self, api_key: str, api_base_url: str) -> None:
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")

    def enabled(self) -> bool:
        return bool(self.api_key)

    def request(self, method: str, path: str, payload: dict | None = None, query: dict | None = None) -> dict | str:
        if not self.api_key:
            raise AdaptApiError("Adapt API key is not configured")
        url = f"{self.api_base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return json.loads(data.decode("utf-8"))
                return data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise AdaptApiError(f"Adapt API returned {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise AdaptApiError(f"Could not reach Adapt API: {error.reason}") from error

    def put_file(self, upload_url: str, file_path: Path) -> None:
        data = file_path.read_bytes()
        request = urllib.request.Request(
            upload_url,
            data=data,
            headers={"Content-Type": "text/csv", "Content-Length": str(len(data))},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(request, timeout=90):
                return
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise AdaptApiError(f"Adapt upload failed {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise AdaptApiError(f"Could not upload to Adapt storage: {error.reason}") from error

    def list_datasets(self) -> dict:
        return self.request("GET", "/datasets", query={"limit": 20})

    def create_file_dataset(self, name: str) -> dict:
        return self.request("POST", "/datasets", {"source": {"type": "file", "file_format": "csv", "name": name}})

    def complete_upload(self, name: str, s3_key: str, file_path: Path) -> dict:
        return self.request(
            "POST",
            "/datasets/upload/complete",
            {
                "file_format": "csv",
                "file_size_bytes": file_path.stat().st_size,
                "name": name,
                "s3_key": s3_key,
            },
        )

    def run_dataset(self, dataset_id: str, estimate: bool = True, max_rows: int | None = None) -> dict:
        payload: dict = {
            "column_mapping": {
                "prompt": "instruction",
                "completion": "response",
                "context": ["context"],
            },
            "estimate": estimate,
            "recipe_specification": {
                "recipes": {
                    "deduplication": True,
                    "prompt_rephrase": True,
                    "reasoning_traces": False,
                }
            },
            "brand_controls": {
                "length": "concise",
                "hallucination_mitigation": False,
                "blueprint": "Generate accurate, concise fraud-risk training examples for bank transaction monitoring. Preserve labels, risk actions, and reason-code discipline.",
            },
        }
        if max_rows:
            payload["job_specification"] = {"max_rows": max_rows}
        return self.request("POST", f"/datasets/{dataset_id}/run", payload)

    def status(self, dataset_id: str) -> dict:
        return self.request("GET", f"/datasets/{dataset_id}/status")

    def download(self, dataset_id: str, target_path: Path) -> dict:
        content = self.request("GET", f"/datasets/{dataset_id}/download", query={"fileFormat": "csv"})
        target_path.write_text(str(content), encoding="utf-8")
        return {"path": str(target_path), "bytes": target_path.stat().st_size}


def build_adapt_instruction_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    rows_written = 0
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as source, ADAPT_INSTRUCTION_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=["instruction", "context", "response"])
        writer.writeheader()
        for record in reader:
            label = (record.get("label") or "").strip()
            if not label:
                continue
            action = action_for_label(label, record)
            reasons = reasons_for_record(record)
            instruction = "Analyze this bank transaction for mule-account and fraud-proceeds risk. Return the label, risk action, and concise reason codes."
            context = json.dumps(
                {
                    "transaction_id": record.get("transaction_id"),
                    "amount": number_or_text(record.get("amount")),
                    "channel": record.get("channel"),
                    "sender_account": record.get("sender_account"),
                    "receiver_account": record.get("receiver_account"),
                    "device_id": record.get("device_id"),
                    "sender_location": record.get("sender_location"),
                    "receiver_location": record.get("receiver_location"),
                    "is_watchlisted_account": number_or_text(record.get("is_watchlisted_account")),
                    "is_watchlisted_device": number_or_text(record.get("is_watchlisted_device")),
                    "is_cyber_alert_match": number_or_text(record.get("is_cyber_alert_match")),
                    "graph_link_count": number_or_text(record.get("graph_link_count")),
                    "shared_device_account_count": number_or_text(record.get("shared_device_account_count")),
                    "funds_moved_within_minutes": number_or_text(record.get("funds_moved_within_minutes")),
                    "cashout_detected": number_or_text(record.get("cashout_detected")),
                    "location_mismatch": number_or_text(record.get("location_mismatch")),
                },
                separators=(",", ":"),
            )
            response = json.dumps(
                {
                    "label": label,
                    "risk_score": number_or_text(record.get("risk_score")),
                    "action": action,
                    "reasons": reasons,
                },
                separators=(",", ":"),
            )
            writer.writerow({"instruction": instruction, "context": context, "response": response})
            rows_written += 1
    return {"path": str(ADAPT_INSTRUCTION_PATH), "rows": rows_written, "bytes": ADAPT_INSTRUCTION_PATH.stat().st_size}


def action_for_label(label: str, record: dict) -> str:
    normalized = label.lower()
    if normalized in {"mule", "fraud"}:
        return "Debit freeze and case escalation"
    if normalized == "review":
        return "Hold transaction for review"
    if normalized == "false_positive":
        return "Allow with monitoring after threshold calibration"
    score = float(record.get("risk_score") or 0)
    return "Step-up authentication" if score >= 48 else "Allow with monitoring"


def reasons_for_record(record: dict) -> list[str]:
    reasons = []
    if record.get("is_cyber_alert_match") == "1":
        reasons.append("Cyber or regulatory alert matched")
    if record.get("is_watchlisted_account") == "1":
        reasons.append("Watchlisted account involved")
    if record.get("is_watchlisted_device") == "1":
        reasons.append("Watchlisted or shared suspicious device detected")
    if float(record.get("amount") or 0) > 90000:
        reasons.append("High-value transfer")
    if record.get("cashout_detected") == "1":
        reasons.append("Rapid cashout or fund pass-through detected")
    if record.get("location_mismatch") == "1":
        reasons.append("Location mismatch across sender and receiver activity")
    if float(record.get("graph_link_count") or 0) >= 5:
        reasons.append("Dense mule-network graph links")
    return reasons or ["Normal behavior with no material risk signal"]


def number_or_text(value: str | None):
    if value is None or value == "":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


ADAPT_CLIENT = AdaptClient(ADAPT_API_KEY, ADAPT_API_BASE_URL)


def upload_adapt_dataset() -> dict:
    built = build_adapt_instruction_dataset()
    name = "adaptive-mule-intelligence-training"
    created = ADAPT_CLIENT.create_file_dataset(name)
    instructions = created.get("upload_instructions") or {}
    upload_url = instructions.get("url")
    s3_key = instructions.get("s3_key")
    if not upload_url or not s3_key:
        raise AdaptApiError(f"Adapt did not return upload instructions: {created}")
    ADAPT_CLIENT.put_file(upload_url, ADAPT_INSTRUCTION_PATH)
    completed = ADAPT_CLIENT.complete_upload(name, s3_key, ADAPT_INSTRUCTION_PATH)
    return {"built": built, "created": created, "completed": completed}


def download_adapt_dataset(dataset_id: str) -> dict:
    return ADAPT_CLIENT.download(dataset_id, ADAPT_DOWNLOAD_PATH)
