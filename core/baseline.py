from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6


def choose_action(obs, step):
    subs = obs.visible_subscriptions

    if not subs:
        return None

    for sub in subs:
        # reveal hidden early
        if getattr(sub, "hidden", False) and step < 2:
            return Action(action_type="investigate", subscription_id=sub.id)

        # cancel expensive or trial subs
        if sub.cost > 1000 or getattr(sub, "trial", False):
            return Action(action_type="cancel", subscription_id=sub.id)

    # default: keep cheapest
    return Action(action_type="keep", subscription_id=subs[0].id)


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
