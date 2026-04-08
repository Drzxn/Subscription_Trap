from models import Observation, Action, Reward
from tasks import get_task
from grader import evaluate_step


class SubscriptionEnv:
    def __init__(self, task_name="hard"):
        self.task_name = task_name
        self.reset()

    def reset(self):
        self.task = get_task(self.task_name)
        self.state = self.task["subscriptions"]
        self.truth = self.task["ground_truth"]
        self.bank_logs = self.task["bank_logs"]
        self.email_logs = self.task["email_logs"]
        self.month = 0
        self.budget = self.task["budget"]
        self.action_count = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        visible = [
            s for s in self.state
            if (not s.hidden or self.month > 1) and s.active
        ]

        return Observation(
            visible_subscriptions=visible,
            bank_logs=self.bank_logs,
            email_logs=self.email_logs,
            month=self.month,
            budget=self.budget,
            action_count=self.action_count
        )

    def step(self, action: Action):
        # 🔒 If already done → freeze environment
        if self.done:
            return self._get_obs(), Reward(value=0, reason="episode_done"), True, {}

        # ⏩ Apply time progression
        self.month += 1
        self.action_count += 1

        # 🎯 Apply action
        for sub in self.state:
            if sub.id == action.subscription_id:
                if action.action_type == "cancel":
                    sub.active = False
                elif action.action_type == "snooze":
                    sub.trial_remaining += 1

        # 🔄 Update trials → convert to paid
        for sub in self.state:
            if sub.trial:
                sub.trial_remaining -= 1
                if sub.trial_remaining <= 0:
                    sub.trial = False

        # 🧠 Compute reward
        reward_value, reason = evaluate_step(
            self.state,
            self.truth,
            self.budget,
            self.action_count
        )

        # 🏁 Check termination AFTER reward
        if self.month >= 6:
            self.done = True

        return (
            self._get_obs(),
            Reward(value=reward_value, reason=reason),
            self.done,
            {}
        )
