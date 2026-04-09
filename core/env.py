from typing import List
from core.models import Observation, Action, Reward
from core.tasks import get_task
from core.grader import evaluate_step
from copy import deepcopy


class SubscriptionEnv:
    def __init__(self, task_name: str = "hard"):
        self.task_name = task_name
        self.reset()

    # 🔹 Reset environment

    def reset(self):
        self.task = get_task(self.task_name)

        # ✅ Safe deep copy
        self.state = deepcopy(self.task.get("subscriptions", []))

        self.truth = self.task.get("ground_truth", [])
        self.bank_logs = self.task.get("bank_logs", [])
        self.email_logs = self.task.get("email_logs", [])

        self.month = 0
        self.budget = 1499.4
        self.action_count = 0
        self.done = False

        return self._get_obs()

    # 🔹 Build observation


def _get_obs(self):
    visible = []

    for s in self.state:
        active = s.get("active", False)
        hidden = s.get("hidden", False)

        # ✅ Better reveal logic
        if active:
            if not hidden:
                visible.append(s)
            elif hidden and self.month >= 2:
                visible.append(s)

    return Observation(
        visible_subscriptions=visible,
        bank_logs=self.bank_logs,
        email_logs=self.email_logs,
        month=self.month,
        budget=round(self.budget, 2),  # ✅ correct place to round
        action_count=self.action_count
    )

    # 🔹 Step function (core RL loop)
    def step(self, action: Action):
        # 🔒 If already finished → freeze
        if self.done:
            return (
                self._get_obs(),
                Reward(value=0.0, reason="episode_done"),
                True,
                {}
            )

        # Validate action safely
        if not action or not hasattr(action, "subscription_id"):
            return (
                self._get_obs(),
                Reward(value=-0.1, reason="invalid_action"),
                False,
                {}
            )

        # ⏩ Time progression
        self.month += 1
        self.action_count += 1

        # 🎯 Apply action
        for sub in self.state:
            if getattr(sub, "id", None) == action.subscription_id:
                if action.action_type == "cancel":
                    sub.active = False

                elif action.action_type == "snooze":
                    if getattr(sub, "trial", False):
                        sub.trial_remaining = getattr(
                            sub, "trial_remaining", 0) + 1

                elif action.action_type == "investigate":
                    # reveal hidden subscription
                    if getattr(sub, "hidden", False):
                        sub.hidden = False

        # 🔄 Update trials → convert to paid
        for sub in self.state:
            if getattr(sub, "trial", False):
                sub.trial_remaining = getattr(sub, "trial_remaining", 0) - 1

                if sub.trial_remaining <= 0:
                    sub.trial = False

        # 🧠 Compute reward (safe fallback)
        try:
            reward_value, reason = evaluate_step(
                self.state,
                self.truth,
                self.budget,
                self.action_count
            )
        except Exception:
            reward_value, reason = -0.2, "grader_error"

        # 🏁 Termination condition
        if self.month >= 6:
            self.done = True

        return (
            self._get_obs(),
            Reward(value=float(reward_value), reason=str(reason)),
            self.done,
            {}
        )
