from typing import List
from copy import deepcopy

from core.models import Observation, Action, Reward
from core.tasks import get_task
from core.grader import evaluate_step


class SubscriptionEnv:
    def __init__(self, task_name: str = "hard"):
        self.task_name = task_name
        self.reset()

    # 🔹 Reset environment
    def reset(self):
        self.task = get_task(self.task_name)

        # ✅ Deep copy (safe for Pydantic objects)
        self.state = deepcopy(self.task.get("subscriptions", []))

        self.truth = self.task.get("ground_truth", {})
        self.bank_logs = self.task.get("bank_logs", [])
        self.email_logs = self.task.get("email_logs", [])

        self.month = 0
        self.budget = 1499.4
        self.action_count = 0
        self.done = False

        return self._get_obs()

    # 🔹 Build observation
    def _get_obs(self):
        visible = [
            s for s in self.state
            if s.active and (not getattr(s, "hidden", False) or self.month >= 2)
        ]

        return Observation(
            visible_subscriptions=visible,
            bank_logs=self.bank_logs,
            email_logs=self.email_logs,
            month=self.month,
            budget=round(self.budget, 2),
            action_count=self.action_count
        )

    # 🔹 Step function (core RL loop)
    def step(self, action: Action):

        # 🔒 Freeze if done
        if self.done:
            return (
                self._get_obs(),
                Reward(value=0.0, reason="episode_done"),
                True,
                {}
            )

        # ❌ Invalid action
        if not action or not hasattr(action, "subscription_id"):
            return (
                self._get_obs(),
                Reward(value=-0.1, reason="invalid_action"),
                False,
                {}
            )

        if action.action_type not in ["cancel", "keep", "snooze", "investigate"]:
            return (
                self._get_obs(),
                Reward(value=-0.1, reason="invalid_action_type"),
                False,
                {}
            )

        # ⏩ Time progression
        self.month += 1
        self.action_count += 1

        # 🎯 Apply action
        for sub in self.state:
            if sub.id == action.subscription_id:

                if action.action_type == "cancel":
                    sub.active = False

                elif action.action_type == "snooze":
                    if getattr(sub, "trial", False):
                        sub.trial_remaining += 1

                elif action.action_type == "investigate":
                    if getattr(sub, "hidden", False):
                        sub.hidden = False

                elif action.action_type == "keep":
                    pass

        # 🔄 Trial progression
        for sub in self.state:
            if getattr(sub, "trial", False):
                sub.trial_remaining -= 1

                if sub.trial_remaining <= 0:
                    sub.trial = False

        # 💰 Update budget (FIXED — CONTROLLED DRAIN)
        monthly_cost = sum(
            s.cost for s in self.state if s.active and not s.trial
        )

        # 🔥 CRITICAL FIX (prevents reward collapse)
        self.budget = max(self.budget - (monthly_cost * 0.2), 0)
        # 🧠 Reward calculation
        try:
            reward_value, reason = evaluate_step(
                self.state,
                self.truth,
                self.budget,
                self.action_count
            )
        except Exception:
            reward_value, reason = -0.2, "grader_error"

        # 🏁 Termination
        if self.month >= 6:
            self.done = True

        # 📊 Info (useful for debugging + evaluation)
        info = {
            "month": self.month,
            "budget": round(self.budget, 2),
            "active_subscriptions": [
                s.id for s in self.state if s.active
            ]
        }

        return (
            self._get_obs(),
            Reward(value=float(round(reward_value, 2)), reason=str(reason)),
            self.done,
            info
        )
