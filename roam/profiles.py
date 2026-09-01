"""Portable profile serialization and example travel personalities."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .data import FEATURES
from .model import PreferenceProfile


def profile_to_dict(name: str, profile: PreferenceProfile) -> dict[str, Any]:
    return {
        "format": "roam-profile-v1",
        "name": name.strip() or "Traveler",
        "features": list(FEATURES),
        "weights": profile.weights.round(8).tolist(),
        "comparisons": profile.comparisons,
    }


def profile_to_json(name: str, profile: PreferenceProfile) -> str:
    return json.dumps(profile_to_dict(name, profile), indent=2)


def profile_from_dict(payload: dict[str, Any]) -> tuple[str, PreferenceProfile]:
    if payload.get("format") != "roam-profile-v1":
        raise ValueError("This is not a Roam profile file.")
    if payload.get("features") != list(FEATURES):
        raise ValueError("The profile uses a different destination feature set.")
    weights = np.asarray(payload.get("weights"), dtype=float)
    if weights.shape != (len(FEATURES),) or not np.all(np.isfinite(weights)):
        raise ValueError("The profile weights are missing or invalid.")
    comparisons = max(0, int(payload.get("comparisons", 0)))
    covariance = np.eye(len(FEATURES)) / max(1.0, comparisons)
    return str(payload.get("name", "Traveler")), PreferenceProfile(weights, covariance, comparisons)


def profile_from_json(raw: str) -> tuple[str, PreferenceProfile]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("The uploaded file is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("The profile must contain a JSON object.")
    return profile_from_dict(payload)


def demo_profiles() -> dict[str, PreferenceProfile]:
    """Return clearly labeled synthetic profiles for trying group mode."""
    presets = {
        "Theo · wild outdoors": {"Nature": 1.8, "Adventure": 1.6, "Cool climate": 1.0, "Nightlife": -.8, "History": -.3},
        "Sam · beach + recharge": {"Beach": 1.8, "Relaxation": 1.7, "Food": .6, "Cool climate": -1.1, "Adventure": -.2},
    }
    result: dict[str, PreferenceProfile] = {}
    for name, preferences in presets.items():
        weights = np.asarray([preferences.get(feature, 0.0) for feature in FEATURES], dtype=float)
        result[name] = PreferenceProfile(weights, np.eye(len(FEATURES)) * .12, 12)
    return result
