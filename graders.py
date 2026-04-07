"""
EduNexora AI — Graders
Computes dynamic rewards for each task based on action correctness.
All rewards are in (0, 1) — FIXED
"""

from typing import Any, Dict, Optional


# 🔥 ADD THIS (NEW)
def clamp(score: float) -> float:
    if score <= 0:
        return 0.001
    elif score >= 1:
        return 0.999
    return round(score, 4)


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 1 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_classification(predicted: str, marks: float) -> float:
    if marks >= 40:
        expected = "pass"
    elif marks >= 35:
        expected = "fail"
    else:
        expected = "backlog"

    score = 1.0 if predicted == expected else 0.0   # CHANGED
    return clamp(score)                            # CHANGED


def grade_ranking(predicted_order: list, actual_marks: Dict[str, float]) -> float:
    if not predicted_order:
        return 0.001   # CHANGED

    sorted_ids = sorted(actual_marks.keys(), key=lambda sid: actual_marks[sid], reverse=True)

    if predicted_order == sorted_ids:
        return 0.5

    top_n = min(3, len(sorted_ids))
    correct = sum(1 for i in range(top_n) if i < len(predicted_order) and predicted_order[i] == sorted_ids[i])

    score = (correct / top_n) * 0.5   # CHANGED
    return clamp(score)               # CHANGED


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 2 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_prioritization(selected_unit: str, lag_scores: Dict[str, int]) -> float:
    if not lag_scores:
        return 0.001   # CHANGED

    best = max(lag_scores, key=lag_scores.get)

    if selected_unit == best:
        return 0.5

    selected_lag = lag_scores.get(selected_unit, 0)
    best_lag = lag_scores[best]

    if best_lag == 0:
        return 0.001   # CHANGED

    score = max(0.2, (selected_lag / best_lag) * 0.5)   # CHANGED
    return clamp(score)                                 # CHANGED


def grade_topic_completion(progress_pct: float) -> float:
    score = progress_pct / 100.0   # CHANGED
    return clamp(score)            # CHANGED


def grade_notification(notifications: list) -> float:
    if not notifications:
        return 0.001   # CHANGED

    keywords = ["lagging", "progress", "class", "complete", "required", "%"]
    hits = sum(1 for n in notifications for kw in keywords if kw.lower() in n.lower())

    score = min(1.0, hits / max(len(notifications) * 2, 1))   # CHANGED
    score = max(0.3, score)                                  # CHANGED

    return clamp(score)                                      # CHANGED


# ──────────────────────────────────────────────────────────────────────────── #
#  Task 3 Graders                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def grade_risk_classification(predicted_risk: str, marks: float) -> float:
    if marks < 40:
        expected = "high"
    elif marks <= 60:
        expected = "medium"
    else:
        expected = "low"

    score = 1.0 if predicted_risk == expected else 0.0   # CHANGED
    return clamp(score)                                 # CHANGED


def grade_intervention(risk_level: str, intervention: str) -> float:
    if not intervention:
        return 0.001   # CHANGED

    severity_keywords = {
        "high":   ["counseling", "remedial", "parent", "immediate"],
        "medium": ["mentoring", "weekly", "practice"],
        "low":    ["monitoring", "enrichment", "optional"],
    }

    expected_kws = severity_keywords.get(risk_level, [])
    hits = sum(1 for kw in expected_kws if kw.lower() in intervention.lower())

    if hits >= 2:
        score = 0.5   # CHANGED
    elif hits == 1:
        score = 0.3   # CHANGED
    else:
        score = 0.1   # CHANGED

    return clamp(score)   # CHANGED


# ──────────────────────────────────────────────────────────────────────────── #
#  Aggregate Grader                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def compute_episode_score(rewards: list) -> Dict[str, Any]:
    if not rewards:
        return {"mean": 0.001, "max": 0.001, "min": 0.001, "total": 0.001, "count": 0}   # CHANGED

    return {
        "mean":  clamp(sum(rewards) / len(rewards)),   # CHANGED
        "max":   clamp(max(rewards)),                  # CHANGED
        "min":   clamp(min(rewards)),                  # CHANGED
        "total": clamp(sum(rewards)),                  # CHANGED
        "count": len(rewards),
    }