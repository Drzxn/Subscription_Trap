from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 12.0  # 6 steps × max ~2 reward


def choose_action(obs):
    """
    Simple rule-based policy:
    - cancel high-cost subscriptions
    - cancel trials
    - otherwise keep
    """

    if not obs.visible_subscriptions:
        return None

    # pick the most expensive or risky subscription
    target = sorted(
        obs.visible_subscriptions,
        key=lambda s: (s.trial, s.cost),
        reverse=True
    )[0]

    if target.trial or target.cost > 1000:
        action_type = "cancel"
    else:
        action_type = "keep"

    return Action(
        action_type=action_type,
        subscription_id=target.id
    )


def run_baseline():
    env = SubscriptionEnv("hard")

    obs = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS):

        action = choose_action(obs)

        # fallback safety
        if action is None:
            break

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)
        total_reward += float(reward_value)

        if done:
            break

    # 🔥 NORMALIZATION (CRITICAL)
    score = total_reward / MAX_TOTAL_REWARD

    # clamp to [0,1]
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
