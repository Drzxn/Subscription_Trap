from copy import deepcopy

from core.models import Observation, Action, Reward
from core.tasks import get_task
from core.grader import evaluate_step


class SubscriptionEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.reset()

    # 🔹 Reset environment
    def reset(self):
        self.task = get_task(self.task_name)

        # Deep copy
        self.state = deepcopy(self.task.get("subscriptions", []))

        self.truth = self.task.get("ground_truth", {})
        self.bank_logs = self.task.get("bank_logs", [])
        self.email_logs = self.task.get("email_logs", [])

        self.month = 0

        # ✅ USE TASK BUDGET (CRITICAL FIX)
        self.budget = self.task.get("budget", 1500.0)

        self.action_count = 0
        self.done = False

        return self._get_obs()

    # 🔹 Observation
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

    # 🔹 Step
    def step(self, action: Action):

        # Freeze if done
        if self.done:
            return self._get_obs(), Reward(value=0.0, reason="done"), True, {}

        # Invalid action
        if not action or not hasattr(action, "subscription_id"):
            return self._get_obs(), Reward(value=-0.1, reason="invalid"), False, {}

        if action.action_type not in ["cancel", "keep", "snooze", "investigate"]:
            return self._get_obs(), Reward(value=-0.1, reason="invalid_type"), False, {}

        # Time update
        self.month += 1
        self.action_count += 1

        # Apply action
        for sub in self.state:
            if sub.id == action.subscription_id:

                if action.action_type == "cancel":
                    sub.active = False

                elif action.action_type == "snooze":
                    if getattr(sub, "trial", False):
                        sub.trial_remaining = getattr(
                            sub, "trial_remaining", 0) + 1

                elif action.action_type == "investigate":
                    if getattr(sub, "hidden", False):
                        sub.hidden = False

                elif action.action_type == "keep":
                    pass  # explicit no-op

        # Trial progression
        for sub in self.state:
            if getattr(sub, "trial", False):
                sub.trial_remaining = getattr(sub, "trial_remaining", 0) - 1

                if sub.trial_remaining <= 0:
                    sub.trial = False

        # Budget update (stable decay)
        monthly_cost = sum(
            s.cost for s in self.state if s.active and not s.trial
        )

        # 🔥 CONTROLLED DRAIN (IMPORTANT FOR REWARD STABILITY)
        self.budget -= monthly_cost * 0.05
        self.budget = max(self.budget, 500)

        # Reward calculation
        try:
            reward_value, reason = evaluate_step(
                self.state,
                self.truth,
                self.budget,
                self.action_count
            )
        except Exception:
            reward_value, reason = -0.2, "grader_error"

        # Termination
        if self.month >= 6:
            self.done = True

        # Info (validator-friendly)
        info = {
            "task": self.task_name,
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
