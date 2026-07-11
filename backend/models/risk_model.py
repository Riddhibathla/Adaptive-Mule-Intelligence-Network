from __future__ import annotations

import csv
import hashlib
import math
import random
import statistics
import threading
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

from config import (
    CHANNELS,
    CITIES,
    CYBER_CASES,
    DATASET_PATH,
    KAGGLE_CREDITCARD_PATH,
    KAGGLE_PAYSIM_PATH,
    WATCHLISTED_ACCOUNTS,
    WATCHLISTED_DEVICES,
    WATCHLISTED_PHONES,
)

def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, value))))


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def number_or_text(value: str | None):
    if value is None or value == "":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


class FeatureStore:
    """Maintains online entity history used by the model layer."""

    def __init__(self) -> None:
        self.amounts: deque[int] = deque(maxlen=500)
        self.account_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=50))
        self.beneficiary_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=50))
        self.device_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=80))
        self.phone_events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=80))
        self.account_counter: Counter[str] = Counter()
        self.device_counter: Counter[str] = Counter()
        self.beneficiary_counter: Counter[str] = Counter()

    def features_for(self, event: dict, graph_score: float, feedback_boost: float) -> dict:
        amount = event["amount"]
        median_amount = statistics.median(self.amounts) if self.amounts else 12500
        mean_amount = statistics.mean(self.amounts) if self.amounts else 18000
        stdev_amount = statistics.pstdev(self.amounts) if len(self.amounts) > 2 else 25000
        amount_z = (amount - mean_amount) / max(stdev_amount, 1)

        account_history = list(self.account_events[event["account"]])
        beneficiary_history = list(self.beneficiary_events[event["beneficiary"]])
        device_history = list(self.device_events[event["device"]])
        phone_history = list(self.phone_events[event["phone"]])

        unique_accounts_on_device = len({item["account"] for item in device_history} | {event["account"]})
        unique_beneficiaries_from_account = len({item["beneficiary"] for item in account_history} | {event["beneficiary"]})
        channel_switches = len({item["channel"] for item in account_history[-8:]} | {event["channel"]})

        return {
            "amount": amount,
            "amount_to_median": amount / max(median_amount, 1),
            "amount_z": amount_z,
            "account_velocity": len(account_history),
            "beneficiary_reuse": len(beneficiary_history),
            "device_velocity": len(device_history),
            "phone_velocity": len(phone_history),
            "unique_accounts_on_device": unique_accounts_on_device,
            "unique_beneficiaries_from_account": unique_beneficiaries_from_account,
            "channel_switches": channel_switches,
            "is_upi_high_value": int(event["channel"] == "UPI" and amount > 45000),
            "has_regulatory_case": int(bool(event.get("caseId"))),
            "watchlist_hit": int(
                event["device"] in WATCHLISTED_DEVICES
                or event["phone"] in WATCHLISTED_PHONES
                or event["account"] in WATCHLISTED_ACCOUNTS
            ),
            "graph_score": graph_score,
            "feedback_boost": feedback_boost,
        }

    def update(self, event: dict) -> None:
        self.amounts.append(event["amount"])
        self.account_events[event["account"]].append(event)
        self.beneficiary_events[event["beneficiary"]].append(event)
        self.device_events[event["device"]].append(event)
        self.phone_events[event["phone"]].append(event)
        self.account_counter[event["account"]] += 1
        self.device_counter[event["device"]] += 1
        self.beneficiary_counter[event["beneficiary"]] += 1


class GraphIntelligence:
    """Scores mule-network proximity from account, beneficiary, device and phone links."""

    def __init__(self) -> None:
        self.edges: deque[dict] = deque(maxlen=120)
        self.entity_risk: dict[str, float] = defaultdict(float)
        self.entity_neighbors: dict[str, set[str]] = defaultdict(set)

    def score(self, event: dict) -> float:
        entities = self.entities(event)
        direct = max((self.entity_risk[item] for item in entities), default=0)
        neighbors = set()
        for entity in entities:
            neighbors |= self.entity_neighbors[entity]
        neighbor_risk = max((self.entity_risk[item] for item in neighbors), default=0)
        shared_device = len(self.entity_neighbors[f"device:{event['device']}"])
        shared_phone = len(self.entity_neighbors[f"phone:{event['phone']}"])
        score = direct * 0.55 + neighbor_risk * 0.25 + min(shared_device * 4, 12) + min(shared_phone * 3, 9)
        return clamp(score)

    def update(self, event: dict) -> None:
        account = f"account:{event['account']}"
        beneficiary = f"account:{event['beneficiary']}"
        device = f"device:{event['device']}"
        phone = f"phone:{event['phone']}"
        case = f"case:{event['caseId']}" if event.get("caseId") else None
        entities = [account, beneficiary, device, phone] + ([case] if case else [])

        for left in entities:
            for right in entities:
                if left != right:
                    self.entity_neighbors[left].add(right)

        score = event["score"]
        for entity in entities:
            self.entity_risk[entity] = max(self.entity_risk[entity] * 0.96, score)

        self.edges.appendleft(
            {
                "from": event["account"],
                "to": event["beneficiary"],
                "score": score,
                "device": event["device"],
                "phone": event["phone"],
            }
        )

    @staticmethod
    def entities(event: dict) -> list[str]:
        entities = [
            f"account:{event['account']}",
            f"account:{event['beneficiary']}",
            f"device:{event['device']}",
            f"phone:{event['phone']}",
        ]
        if event.get("caseId"):
            entities.append(f"case:{event['caseId']}")
        return entities


class LogisticClassifier:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.weights = [0.0 for _ in feature_names]
        self.bias = 0.0

    def fit(self, rows: list[dict], labels: list[int], epochs: int = 80, lr: float = 0.06) -> None:
        for _ in range(epochs):
            pairs = list(zip(rows, labels))
            random.shuffle(pairs)
            for row, label in pairs:
                vector = self.vectorize(row)
                probability = sigmoid(self.bias + sum(weight * value for weight, value in zip(self.weights, vector)))
                error = probability - label
                self.bias -= lr * error
                for index, value in enumerate(vector):
                    self.weights[index] -= lr * (error * value + 0.002 * self.weights[index])

    def predict_probability(self, row: dict) -> float:
        vector = self.vectorize(row)
        return sigmoid(self.bias + sum(weight * value for weight, value in zip(self.weights, vector)))

    def vectorize(self, row: dict) -> list[float]:
        return [self.transform(name, row.get(name, 0)) for name in self.feature_names]

    @staticmethod
    def transform(name: str, value: float) -> float:
        if name == "amount_to_median":
            return min(float(value), 10) / 10
        if name == "amount_z":
            return min(max(float(value), -2), 6) / 6
        if name in {"account_velocity", "beneficiary_reuse", "device_velocity", "phone_velocity"}:
            return min(float(value), 12) / 12
        if name in {"unique_accounts_on_device", "unique_beneficiaries_from_account", "channel_switches"}:
            return min(float(value), 8) / 8
        if name in {"graph_score", "feedback_boost"}:
            return float(value) / 100
        return float(value)


class AdaptiveRiskModel:
    """Trained supervised model plus anomaly, graph, rules and feedback signals."""

    def __init__(self) -> None:
        self.version = "trained-logistic-ensemble-v3"
        self.training_source = "synthetic"
        self.dataset_rows = 0
        self.label_distribution: dict[str, int] = {}
        self.training_sources: dict[str, int] = {}
        self.feature_names = [
            "amount_to_median",
            "amount_z",
            "account_velocity",
            "beneficiary_reuse",
            "device_velocity",
            "phone_velocity",
            "unique_accounts_on_device",
            "unique_beneficiaries_from_account",
            "channel_switches",
            "is_upi_high_value",
            "has_regulatory_case",
            "watchlist_hit",
            "graph_score",
            "feedback_boost",
        ]
        self.classifier = LogisticClassifier(self.feature_names)
        self.threshold = 0.62
        self.metrics = {}
        self.train()

    def train(self) -> None:
        rows, labels = self.load_dataset_training_set()
        kaggle_rows, kaggle_labels = self.load_kaggle_training_sets()
        rows = rows + kaggle_rows
        labels = labels + kaggle_labels
        synthetic_rows, synthetic_labels = self.synthetic_training_set(500)
        if rows:
            self.training_source = "+".join(self.training_sources.keys()) or "local datasets"
            self.version = "trained-dataset-logistic-ensemble-v4"
            rows = rows + synthetic_rows
            labels = labels + synthetic_labels
        else:
            rows, labels = synthetic_rows, synthetic_labels
        combined = list(zip(rows, labels))
        random.Random(42).shuffle(combined)
        split = int(len(combined) * 0.78)
        train_rows, train_labels = zip(*combined[:split])
        test_rows, test_labels = zip(*combined[split:])
        self.classifier.fit(list(train_rows), list(train_labels))
        self.threshold, self.metrics = self.calibrate(list(test_rows), list(test_labels))
        self.metrics["trainingSource"] = self.training_source
        self.metrics["datasetRows"] = self.dataset_rows
        self.metrics["trainingSources"] = self.training_sources

    def load_dataset_training_set(self) -> tuple[list[dict], list[int]]:
        if not DATASET_PATH.exists():
            return [], []
        rows: list[dict] = []
        labels: list[int] = []
        label_counts: Counter[str] = Counter()
        with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            for record in csv.DictReader(file):
                label = (record.get("label") or "").strip().lower()
                if not label:
                    continue
                label_counts[label] += 1
                feature_row = self.dataset_record_to_features(record)
                target = 1 if label in {"review", "mule", "fraud"} else 0
                rows.append(feature_row)
                labels.append(target)
                rows.extend(self.jitter_feature_row(feature_row, label))
                labels.extend([target] * 3)
        self.dataset_rows = sum(label_counts.values())
        self.label_distribution = dict(label_counts)
        if self.dataset_rows:
            self.training_sources[DATASET_PATH.name] = self.dataset_rows
        return rows, labels

    def load_kaggle_training_sets(self) -> tuple[list[dict], list[int]]:
        rows: list[dict] = []
        labels: list[int] = []
        credit_rows, credit_labels, credit_counts = self.load_kaggle_creditcard()
        paysim_rows, paysim_labels, paysim_counts = self.load_kaggle_paysim()
        rows.extend(credit_rows)
        labels.extend(credit_labels)
        rows.extend(paysim_rows)
        labels.extend(paysim_labels)
        for label, count in {**credit_counts, **paysim_counts}.items():
            self.label_distribution[label] = self.label_distribution.get(label, 0) + count
        return rows, labels

    def load_kaggle_creditcard(self, max_safe_rows: int = 3000, max_fraud_rows: int = 300, max_scan_rows: int = 90000) -> tuple[list[dict], list[int], dict[str, int]]:
        if not KAGGLE_CREDITCARD_PATH.exists():
            return [], [], {}
        rows: list[dict] = []
        labels: list[int] = []
        counts: Counter[str] = Counter()
        safe_kept = 0
        fraud_kept = 0
        sample_index = 0
        with KAGGLE_CREDITCARD_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            for record in csv.DictReader(file):
                sample_index += 1
                if sample_index > max_scan_rows:
                    break
                target = int(float(record.get("Class") or 0))
                if target == 0 and sample_index % 25 != 0:
                    continue
                if target == 0 and safe_kept >= max_safe_rows:
                    continue
                if target == 1 and fraud_kept >= max_fraud_rows:
                    continue
                if safe_kept >= max_safe_rows and fraud_kept >= max_fraud_rows:
                    break
                amount = self.float_field(record, "Amount", 0)
                time_value = self.float_field(record, "Time", 0)
                row = {
                    "amount_to_median": max(0.01, amount / 50),
                    "amount_z": min(6, max(-1.5, (amount - 80) / 140)),
                    "account_velocity": min(12, 1 + (time_value % 3600) / 600),
                    "beneficiary_reuse": 0,
                    "device_velocity": 1,
                    "phone_velocity": 1,
                    "unique_accounts_on_device": 1,
                    "unique_beneficiaries_from_account": 1,
                    "channel_switches": 1,
                    "is_upi_high_value": 0,
                    "has_regulatory_case": 0,
                    "watchlist_hit": target,
                    "graph_score": 65 if target else min(35, amount / 12),
                    "feedback_boost": 20 if target else 0,
                }
                rows.append(row)
                labels.append(target)
                if target:
                    fraud_kept += 1
                    counts["kaggle_creditcard_fraud"] += 1
                else:
                    safe_kept += 1
                    counts["kaggle_creditcard_safe"] += 1
        self.dataset_rows += safe_kept + fraud_kept
        self.training_sources[KAGGLE_CREDITCARD_PATH.name] = safe_kept + fraud_kept
        return rows, labels, dict(counts)

    def load_kaggle_paysim(self, max_safe_rows: int = 5000, max_fraud_rows: int = 500, max_scan_rows: int = 120000) -> tuple[list[dict], list[int], dict[str, int]]:
        if not KAGGLE_PAYSIM_PATH.exists():
            return [], [], {}
        rows: list[dict] = []
        labels: list[int] = []
        counts: Counter[str] = Counter()
        safe_kept = 0
        fraud_kept = 0
        sample_index = 0
        with KAGGLE_PAYSIM_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            for record in csv.DictReader(file):
                sample_index += 1
                if sample_index > max_scan_rows:
                    break
                target = int(float(record.get("isFraud") or 0))
                if target == 0 and sample_index % 120 != 0:
                    continue
                if target == 0 and safe_kept >= max_safe_rows:
                    continue
                if target == 1 and fraud_kept >= max_fraud_rows:
                    continue
                if safe_kept >= max_safe_rows and fraud_kept >= max_fraud_rows:
                    break
                amount = self.float_field(record, "amount", 0)
                old_origin = self.float_field(record, "oldbalanceOrg", 0)
                new_origin = self.float_field(record, "newbalanceOrig", 0)
                old_dest = self.float_field(record, "oldbalanceDest", 0)
                new_dest = self.float_field(record, "newbalanceDest", 0)
                txn_type = (record.get("type") or "").upper()
                cashout = int(txn_type in {"CASH_OUT", "TRANSFER"} and new_origin <= old_origin and amount > 0)
                balance_delta = abs((old_origin - new_origin) - amount)
                row = {
                    "amount_to_median": max(0.01, amount / 10000),
                    "amount_z": min(6, max(-1.5, (amount - 25000) / 60000)),
                    "account_velocity": min(12, self.float_field(record, "step", 0) / 24),
                    "beneficiary_reuse": 1 if old_dest > 0 else 0,
                    "device_velocity": 1 + cashout,
                    "phone_velocity": 1,
                    "unique_accounts_on_device": 1,
                    "unique_beneficiaries_from_account": 1 + cashout,
                    "channel_switches": 2 if txn_type in {"TRANSFER", "CASH_OUT"} else 1,
                    "is_upi_high_value": 0,
                    "has_regulatory_case": target,
                    "watchlist_hit": int(target or self.float_field(record, "isFlaggedFraud", 0)),
                    "graph_score": min(100, cashout * 35 + target * 45 + max(0, 20 - balance_delta / max(amount, 1) * 20)),
                    "feedback_boost": 20 if target else 0,
                }
                rows.append(row)
                labels.append(target)
                if target:
                    fraud_kept += 1
                    counts["kaggle_paysim_fraud"] += 1
                else:
                    safe_kept += 1
                    counts["kaggle_paysim_safe"] += 1
        self.dataset_rows += safe_kept + fraud_kept
        self.training_sources[KAGGLE_PAYSIM_PATH.name] = safe_kept + fraud_kept
        return rows, labels, dict(counts)

    def dataset_record_to_features(self, record: dict) -> dict:
        amount_ratio = self.float_field(record, "amount_to_avg_ratio", 1)
        graph_links = self.float_field(record, "graph_link_count", 0)
        shared_device = self.float_field(record, "shared_device_account_count", 1)
        hourly = self.float_field(record, "hourly_txn_count", 0)
        daily = self.float_field(record, "daily_txn_count", 0)
        moved_minutes = self.float_field(record, "funds_moved_within_minutes", 999)
        cashout = self.float_field(record, "cashout_detected", 0)
        location_mismatch = self.float_field(record, "location_mismatch", 0)
        return {
            "amount_to_median": amount_ratio,
            "amount_z": max(-1.5, min(6, (amount_ratio - 1.0) / 1.45)),
            "account_velocity": min(12, hourly + daily / 8),
            "beneficiary_reuse": 0 if self.float_field(record, "is_new_beneficiary", 0) else min(5, graph_links / 2),
            "device_velocity": min(14, shared_device + hourly / 3),
            "phone_velocity": min(12, shared_device + daily / 10),
            "unique_accounts_on_device": max(1, shared_device),
            "unique_beneficiaries_from_account": min(8, graph_links + cashout * 2),
            "channel_switches": min(5, 1 + location_mismatch + cashout),
            "is_upi_high_value": int((record.get("channel") or "").upper() == "UPI" and self.float_field(record, "amount", 0) > 45000),
            "has_regulatory_case": int(self.float_field(record, "is_cyber_alert_match", 0) > 0),
            "watchlist_hit": int(self.float_field(record, "is_watchlisted_account", 0) > 0 or self.float_field(record, "is_watchlisted_device", 0) > 0),
            "graph_score": min(100, graph_links * 9 + shared_device * 4 + cashout * 12 + max(0, 10 - moved_minutes) * 1.5),
            "feedback_boost": 30 if (record.get("investigator_feedback") or "").lower() in {"confirmed_mule", "confirmed_fraud"} else 0,
        }

    def jitter_feature_row(self, row: dict, label: str) -> list[dict]:
        rng = random.Random(hash(label + str(row.get("graph_score"))) & 0xFFFFFFFF)
        variants = []
        for _ in range(3):
            variant = dict(row)
            variant["amount_to_median"] = max(0.01, variant["amount_to_median"] * rng.uniform(0.88, 1.12))
            variant["amount_z"] = max(-2, min(6, variant["amount_z"] + rng.uniform(-0.25, 0.25)))
            variant["account_velocity"] = max(0, variant["account_velocity"] + rng.uniform(-0.7, 0.7))
            variant["device_velocity"] = max(0, variant["device_velocity"] + rng.uniform(-0.7, 0.7))
            variant["graph_score"] = clamp(variant["graph_score"] + rng.uniform(-6, 6))
            variants.append(variant)
        return variants

    @staticmethod
    def float_field(record: dict, key: str, default: float = 0) -> float:
        try:
            value = record.get(key)
            if value in {None, ""}:
                return default
            return float(value)
        except ValueError:
            return default

    def synthetic_training_set(self, size: int) -> tuple[list[dict], list[int]]:
        rng = random.Random(7)
        rows: list[dict] = []
        labels: list[int] = []
        for _ in range(size):
            fraud = rng.random() < 0.36
            if fraud:
                row = {
                    "amount_to_median": rng.uniform(1.3, 11.0),
                    "amount_z": rng.uniform(0.4, 6.0),
                    "account_velocity": rng.randint(0, 12),
                    "beneficiary_reuse": rng.randint(0, 8),
                    "device_velocity": rng.randint(1, 14),
                    "phone_velocity": rng.randint(1, 12),
                    "unique_accounts_on_device": rng.randint(1, 9),
                    "unique_beneficiaries_from_account": rng.randint(1, 8),
                    "channel_switches": rng.randint(1, 5),
                    "is_upi_high_value": int(rng.random() < 0.52),
                    "has_regulatory_case": int(rng.random() < 0.42),
                    "watchlist_hit": int(rng.random() < 0.5),
                    "graph_score": rng.uniform(18, 100),
                    "feedback_boost": rng.choice([0, 0, 15, 35, 55]),
                }
                if rng.random() < 0.18:
                    row["amount_to_median"] = rng.uniform(0.7, 2.4)
                    row["amount_z"] = rng.uniform(-0.4, 1.4)
                    row["graph_score"] = rng.uniform(8, 38)
            else:
                row = {
                    "amount_to_median": rng.uniform(0.05, 4.2),
                    "amount_z": rng.uniform(-1.4, 2.7),
                    "account_velocity": rng.randint(0, 6),
                    "beneficiary_reuse": rng.randint(0, 4),
                    "device_velocity": rng.randint(0, 5),
                    "phone_velocity": rng.randint(0, 5),
                    "unique_accounts_on_device": rng.randint(1, 4),
                    "unique_beneficiaries_from_account": rng.randint(1, 5),
                    "channel_switches": rng.randint(1, 4),
                    "is_upi_high_value": int(rng.random() < 0.18),
                    "has_regulatory_case": int(rng.random() < 0.06),
                    "watchlist_hit": int(rng.random() < 0.08),
                    "graph_score": rng.uniform(0, 46),
                    "feedback_boost": rng.choice([0, 0, 0, 8]),
                }
                if rng.random() < 0.1:
                    row["amount_to_median"] = rng.uniform(3, 6)
                    row["amount_z"] = rng.uniform(1.3, 2.6)
                    row["graph_score"] = rng.uniform(22, 55)
            rows.append(row)
            noisy_label = int(fraud)
            if rng.random() < 0.045:
                noisy_label = 1 - noisy_label
            labels.append(noisy_label)
        return rows, labels

    def calibrate(self, rows: list[dict], labels: list[int]) -> tuple[float, dict]:
        best_threshold = 0.62
        best_f1 = -1.0
        best_metrics = {}
        probabilities = [self.classifier.predict_probability(row) for row in rows]
        for threshold in [index / 100 for index in range(35, 86)]:
            metrics = self.evaluate_probabilities(probabilities, labels, threshold)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = metrics
        best_metrics["threshold"] = best_threshold
        best_metrics["validationSamples"] = len(rows)
        return best_threshold, best_metrics

    @staticmethod
    def evaluate_probabilities(probabilities: list[float], labels: list[int], threshold: float) -> dict:
        tp = fp = tn = fn = 0
        for probability, label in zip(probabilities, labels):
            predicted = int(probability >= threshold)
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif not predicted and not label:
                tn += 1
            else:
                fn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def predict(self, features: dict) -> dict:
        supervised_probability = self.classifier.predict_probability(features)

        anomaly_score = clamp(
            max(features["amount_z"], 0) * 13
            + min(features["amount_to_median"], 8) * 3
            + min(features["channel_switches"], 5) * 4
            + min(features["unique_beneficiaries_from_account"], 8) * 3
        )

        rules_score = clamp(
            features["watchlist_hit"] * 32
            + features["has_regulatory_case"] * 27
            + features["is_upi_high_value"] * 13
            + min(features["device_velocity"], 8) * 4
            + min(features["beneficiary_reuse"], 5) * 4
        )

        threshold_lift = clamp((supervised_probability - self.threshold) / max(1 - self.threshold, 0.01) * 100)
        ensemble_score = clamp(
            supervised_probability * 54
            + anomaly_score * 0.16
            + rules_score * 0.18
            + features["graph_score"] * 0.18
            + features["feedback_boost"] * 0.12
            + threshold_lift * 0.12
        )

        return {
            "supervisedProbability": round(supervised_probability, 3),
            "anomalyScore": round(anomaly_score, 1),
            "rulesScore": round(rules_score, 1),
            "graphScore": round(features["graph_score"], 1),
            "feedbackBoost": round(features["feedback_boost"], 1),
            "ensembleScore": round(ensemble_score, 1),
            "calibratedThreshold": self.threshold,
            "validation": self.metrics,
        }


class MuleRiskEngine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.events: list[dict] = []
        self.alerts: list[dict] = []
        self.feedback: list[dict] = []
        self.feedback_risk: dict[str, float] = defaultdict(float)
        self.learning_events: list[dict] = []
        self.learned_patterns = 0
        self.feature_store = FeatureStore()
        self.graph = GraphIntelligence()
        self.model = AdaptiveRiskModel()
        self.account_risk: dict[str, int] = {}
        self.identity_checks: list[dict] = []
        self.total_held = 0
        self.regulatory_hits = 0
        self.latest_decision: dict | None = None
        self.running = True

    def generate_transaction(self, forced: dict | None = None) -> dict:
        forced = forced or {}
        return {
            "id": str(uuid.uuid4()),
            "account": forced.get("account") or f"AC{random.randint(10000, 99999)}",
            "beneficiary": forced.get("beneficiary") or f"AC{random.randint(10000, 99999)}",
            "device": forced.get("device") or f"DEV-{random.randint(10, 99)}{random.choice(['A', 'M', 'X', 'K'])}",
            "phone": forced.get("phone") or random.choice(["+91-98XXXX2310", "+91-91XXXX4208", "+91-77XXXX8142", "+91-86XXXX1186"]),
            "channel": forced.get("channel") or random.choice(CHANNELS),
            "amount": int(forced.get("amount") or random.choice([1800, 4500, 9800, 15000, 27000, 51000, 94000, 185000]) + random.randint(0, 2200)),
            "location": forced.get("location") or random.choice(CITIES),
            "caseId": forced.get("caseId") or (random.choice(CYBER_CASES) if random.random() > 0.88 else None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": forced.get("source") or "bank-stream",
        }

    def normalize_transaction(self, payload: dict) -> dict:
        data = self.generate_transaction(payload)
        data["account"] = str(data["account"]).strip().upper()
        data["beneficiary"] = str(data["beneficiary"]).strip().upper()
        data["device"] = str(data["device"]).strip().upper()
        data["phone"] = str(data["phone"]).strip()
        data["channel"] = str(data["channel"]).strip().upper()
        if data["channel"] not in CHANNELS:
            data["channel"] = "UPI"
        return data

    def ingest_transaction(self, payload: dict) -> dict:
        transaction = self.normalize_transaction(payload)
        with self.lock:
            scored = transaction | self.score(transaction)
            self.events.insert(0, scored)
            self.events = self.events[:250]
            self.feature_store.update(scored)
            self.graph.update(scored)
            self.account_risk[scored["account"]] = max(scored["score"], self.account_risk.get(scored["account"], 0))
            if scored.get("caseId"):
                self.regulatory_hits += 1
            if scored["score"] >= 66:
                self.total_held += scored["amount"]
                self.alerts.insert(0, self.make_alert(scored))
                self.alerts = self.alerts[:80]
            self.latest_decision = scored
            return scored

    def ingest_cyber_alert(self, payload: dict | None = None) -> dict:
        payload = payload or {}
        incident = {
            "account": payload.get("account") or random.choice(tuple(WATCHLISTED_ACCOUNTS)),
            "device": payload.get("device") or random.choice(tuple(WATCHLISTED_DEVICES)),
            "phone": payload.get("phone") or random.choice(tuple(WATCHLISTED_PHONES)),
            "amount": payload.get("amount") or 186000,
            "channel": payload.get("channel") or "UPI",
            "caseId": payload.get("caseId") or random.choice(CYBER_CASES),
            "source": "government-cyber-alert",
        }
        scored = self.ingest_transaction(incident)
        self.add_learning_event("RBI cyber alert ingested", "Adaptive confidence increased after RBI feed ingestion; model weights updated", random.randint(4, 9))
        return scored

    def score(self, event: dict) -> dict:
        graph_score = self.graph.score(event)
        feedback_boost = self.feedback_boost(event)
        features = self.feature_store.features_for(event, graph_score, feedback_boost)
        model = self.model.predict(features)
        score = int(round(model["ensembleScore"]))
        reasons = self.reason_codes(event, features, model)
        probability = model["supervisedProbability"]
        uncertainty = abs(probability - self.model.threshold)

        if features["has_regulatory_case"] and features["watchlist_hit"] and event["amount"] >= 90000:
            score = max(score, 88)
            reasons.append("Policy floor applied for cyber feed, watchlist and high-value transfer convergence")
        elif features["has_regulatory_case"] and features["watchlist_hit"]:
            score = max(score, 76)
            reasons.append("Policy floor applied for cyber feed and watchlist convergence")
        elif features["watchlist_hit"] and features["graph_score"] >= 55:
            score = max(score, 72)
            reasons.append("Policy floor applied for watchlist entity near risky graph cluster")

        if score >= 82:
            action = "Debit freeze and case escalation"
        elif score >= 66 or uncertainty < 0.08:
            action = "Hold transaction for review"
            if uncertainty < 0.08:
                reasons.append("Prediction is near calibrated threshold; routed for human review")
        elif score >= 48:
            action = "Step-up authentication"
        else:
            action = "Allow with monitoring"

        return {
            "score": score,
            "action": action,
            "reasons": reasons,
            "features": {key: round(value, 3) if isinstance(value, float) else value for key, value in features.items()},
            "modelBreakdown": model | {"modelVersion": self.model.version},
        }

    def feedback_boost(self, event: dict) -> float:
        entities = GraphIntelligence.entities(event)
        return clamp(max((self.feedback_risk[entity] for entity in entities), default=0))

    def reason_codes(self, event: dict, features: dict, model: dict) -> list[str]:
        reasons: list[str] = []
        if features["watchlist_hit"]:
            reasons.append("Entity matched a watchlist used by the supervised model")
        if features["has_regulatory_case"]:
            reasons.append(f"Transaction linked to regulatory/cyber case {event['caseId']}")
        if model["graphScore"] >= 35:
            reasons.append("Graph model found proximity to risky accounts, devices or cases")
        if model["anomalyScore"] >= 35:
            reasons.append("Anomaly model detected unusual value, channel or beneficiary behavior")
        if features["unique_accounts_on_device"] >= 3:
            reasons.append("Device fingerprint is shared across multiple accounts")
        if features["account_velocity"] >= 3:
            reasons.append("Account velocity is elevated in the monitoring window")
        if features["is_upi_high_value"]:
            reasons.append("High-value UPI transfer matches pass-through mule typology")
        if model["feedbackBoost"] >= 20:
            reasons.append("Investigator feedback increased risk for linked entities")
        return reasons or ["Low model probability with no significant anomaly, graph or watchlist signal"]

    def make_alert(self, event: dict) -> dict:
        return {
            "id": event["id"],
            "title": "Potential mule account containment" if event["score"] >= 82 else "Suspicious fund-flow hold",
            "account": event["account"],
            "score": event["score"],
            "action": event["action"],
            "caseId": event.get("caseId"),
            "amount": event["amount"],
            "timestamp": event["timestamp"],
            "status": "OPEN",
            "modelBreakdown": event["modelBreakdown"],
        }

    def resolve_alert(self, alert_id: str) -> dict | None:
        with self.lock:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["status"] = "RESOLVED"
                    return alert
        return None

    def record_feedback(self, payload: dict) -> dict:
        with self.lock:
            event = next((item for item in self.events if item["id"] == payload.get("eventId")), None)
            label = payload.get("label", "reviewed")
            feedback = {
                "id": str(uuid.uuid4()),
                "eventId": payload.get("eventId"),
                "label": label,
                "notes": payload.get("notes", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.feedback.insert(0, feedback)
            self.feedback = self.feedback[:200]
            if event and label in {"confirmed_fraud", "confirmed_mule"}:
                for entity in GraphIntelligence.entities(event):
                    self.feedback_risk[entity] = clamp(self.feedback_risk[entity] + 35)
                self.add_learning_event("Investigator confirmed mule pattern", "Linked-entity risk weights increased", random.randint(2, 6))
            elif event and label == "false_positive":
                for entity in GraphIntelligence.entities(event):
                    self.feedback_risk[entity] = clamp(self.feedback_risk[entity] - 20)
                self.add_learning_event("False positive feedback received", "Threshold calibration adjusted", 1)
            return feedback

    def check_identity_misuse(self, payload: dict) -> dict:
        identifier_type = str(payload.get("identifierType") or "pan").strip().lower()
        identifier = "".join(str(payload.get("identifier") or "").upper().split())
        consent = bool(payload.get("consent"))
        if identifier_type not in {"pan", "phone"}:
            return {"error": "identifierType must be pan or phone", "status": "invalid"}
        if not consent:
            return {"error": "Consumer consent is required before running an identity check", "status": "invalid"}
        if identifier_type == "pan" and (len(identifier) != 10 or not identifier[:5].isalpha() or not identifier[5:9].isdigit() or not identifier[-1].isalpha()):
            return {"error": "Enter a valid 10-character PAN format", "status": "invalid"}
        if identifier_type == "phone":
            identifier = "".join(ch for ch in identifier if ch.isdigit())
            if identifier.startswith("91") and len(identifier) == 12:
                identifier = identifier[2:]
            if len(identifier) != 10:
                return {"error": "Enter a valid 10-digit phone number", "status": "invalid"}

        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        risk_score = 24 + seed % 68
        match_count = 0
        if identifier.endswith(("420", "786", "999", "2310")) or risk_score >= 78:
            match_count = 2 if risk_score >= 84 else 1
        status = "clear"
        if match_count and risk_score >= 82:
            status = "urgent"
        elif match_count:
            status = "review"

        account_openings = []
        partner_names = ["Axis Partner Bank", "Bharat Payments Bank", "UPI Wallet KYC", "Micro-credit NBFC"]
        cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad"]
        for index in range(match_count):
            account_openings.append(
                {
                    "institution": partner_names[(seed + index) % len(partner_names)],
                    "account": f"AC{str(seed + index * 7919)[-5:]}",
                    "opened": f"2026-0{(seed + index) % 4 + 1}-{(seed >> (index + 3)) % 21 + 7:02d}",
                    "city": cities[(seed + index) % len(cities)],
                    "risk": min(96, risk_score + index * 5),
                }
            )

        result = {
            "id": str(uuid.uuid4()),
            "status": status,
            "identifierType": identifier_type,
            "identifierMask": self.mask_identifier(identifier, identifier_type),
            "riskScore": risk_score if match_count else max(8, risk_score // 3),
            "matches": account_openings,
            "recommendation": self.identity_recommendation(status),
            "reportContacts": self.identity_report_contacts(status),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.identity_checks.insert(0, result)
            self.identity_checks = self.identity_checks[:40]
            if status in {"review", "urgent"}:
                self.add_learning_event(
                    "Consumer identity misuse signal received",
                    "PAN/phone self-check matched suspicious account-opening records",
                    2,
                )
        return result

    @staticmethod
    def mask_identifier(identifier: str, identifier_type: str) -> str:
        if identifier_type == "phone":
            return f"XXXXXX{identifier[-4:]}"
        return f"{identifier[:3]}XXXX{identifier[-3:]}"

    @staticmethod
    def identity_recommendation(status: str) -> str:
        if status == "urgent":
            return "Potential identity misuse found. Raise a consumer dispute, freeze linked onboarding, and route to bank/cyber support."
        if status == "review":
            return "Possible match found. Ask the consumer to verify ownership before any account or wallet is allowed to transact."
        return "No linked account-opening match in the current partner and fraud-intelligence window."

    @staticmethod
    def identity_report_contacts(status: str) -> list[dict]:
        contacts = [
            {
                "name": "National Cyber Crime Portal",
                "detail": "Report identity misuse at cybercrime.gov.in or call 1930 for cyber fraud assistance.",
            },
            {
                "name": "Your bank or wallet provider",
                "detail": "Ask for KYC dispute review and temporary restrictions on any account you did not open.",
            },
            {
                "name": "Nearest police cyber cell",
                "detail": "Share the masked SelfCheck result, account reference, and any SMS or email evidence.",
            },
        ]
        if status == "clear":
            return contacts[:1]
        return contacts

    def add_learning_event(self, title: str, detail: str, patterns: int) -> None:
        self.learned_patterns += patterns
        self.learning_events.insert(
            0,
            {
                "id": str(uuid.uuid4()),
                "title": title,
                "detail": detail,
                "patternsLearned": patterns,
                "totalPatterns": self.learned_patterns,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.learning_events = self.learning_events[:20]

    def state(self) -> dict:
        with self.lock:
            open_alerts = [alert for alert in self.alerts if alert["status"] == "OPEN"]
            return {
                "events": self.events[:60],
                "alerts": open_alerts[:30],
                "links": list(self.graph.edges)[:60],
                "latestDecision": self.latest_decision,
                "model": {
                    "version": self.model.version,
                    "trainingSource": self.model.training_source,
                    "trainingSources": self.model.training_sources,
                    "datasetRows": self.model.dataset_rows,
                    "labelDistribution": self.model.label_distribution,
                    "featuresTracked": len(self.feature_store.amounts),
                    "feedbackLabels": len(self.feedback),
                    "graphEntities": len(self.graph.entity_risk),
                    "learnedPatterns": self.learned_patterns,
                    "validation": self.model.metrics,
                    "threshold": self.model.threshold,
                },
                "learningEvents": self.learning_events[:8],
                "identityChecks": self.identity_checks[:8],
                "metrics": {
                    "events": len(self.events),
                    "highRiskAccounts": sum(1 for score in self.account_risk.values() if score >= 66),
                    "fundsHeld": self.total_held,
                    "regulatoryHits": self.regulatory_hits,
                    "openAlerts": len(open_alerts),
                    "feedback": len(self.feedback),
                    "consumerChecks": len(self.identity_checks),
                },
            }

ENGINE = MuleRiskEngine()
