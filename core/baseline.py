from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 6.0  # expected upper bound


def choose_action(obs, step):
    """
    Simple but stable heuristic baseline:
    - cancel high-cost subs
    - cancel trials
    - investigate early
    """

    subs = obs.visible_subscriptions

    if not subs:
        return None

    for sub in subs:
        # 🔍 reveal hidden early
        if getattr(sub, "hidden", False) and step < 2:
            return Action(action_type="investigate", subscription_id=sub.id)

        # ❌ cancel expensive or trials
        if sub.cost > 1000 or getattr(sub, "trial", False):
            return Action(action_type="cancel", subscription_id=sub.id)

    # ✅ otherwise keep cheapest
    return Action(action_type="keep", subscription_id=subs[0].id)


def run_baseline():
    env = SubscriptionEnv("hard")

    obs = env.reset()

    total_reward = 0.0
    steps_taken = 0

    for step in range(MAX_STEPS):
        action = choose_action(obs, step)

        if action is None:
            break

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)
        total_reward += float(reward_value)

        steps_taken += 1

        if done:
            break

    # 🔥 FINAL NORMALIZATION (CORRECT + STABLE)
    score = (total_reward + MAX_TOTAL_REWARD) / (2 * MAX_TOTAL_REWARD)

    # clamp to [0,1]
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
