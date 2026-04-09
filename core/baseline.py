from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6


def choose_action(obs, step):
    subs = obs.visible_subscriptions

    if not subs:
        return None

    # Step 1: investigate early
    if step == 0:
        for sub in subs:
            if getattr(sub, "hidden", False):
                return Action("investigate", sub.id)

    for sub in subs:
        # cancel expensive or risky
        if sub.cost > 800 or getattr(sub, "trial", False):
            return Action("cancel", sub.id)

    # otherwise keep cheapest
    return Action("keep", subs[0].id)


def run_baseline():
    env = SubscriptionEnv("hard")

    obs = env.reset()

    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = choose_action(obs, step)

        if action is None:
            break

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)
        total_reward += float(reward_value)

        if done:
            break

    # 🔥 FINAL NORMALIZATION (FIXED RANGE)
    score = (total_reward + 8.0) / 16.0

    # clamp
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
