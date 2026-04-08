# core/__init__.py

"""
Core package for Subscription Trap OpenEnv environment.

Contains:
- Environment logic
- Models (Action, Observation, Reward)
- Baseline agent
- Tasks and graders
"""

# Expose key components for clean imports
from .env import SubscriptionEnv
from .models import Action
from .baseline import run_baseline

__all__ = [
    "SubscriptionEnv",
    "Action",
    "run_baseline",
]
