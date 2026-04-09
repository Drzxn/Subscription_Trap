from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 12.0


def choose_action(obs, step):
    subs = obs.visible_subscriptions

    if not subs:
        return None

    # 🔥 STEP 0–1: DO NOTHING (avoid penalties)
    if step < 2:
        return Action(
            action_type="keep",
            subscription_id=subs[0].id
        )

    # 🔥 STEP 2+: careful decisions

    # cancel only trials first (safe)
    for s in subs:
        if getattr(s, "trial", False):
            return Action(action_type="cancel", subscription_id=s.id)

    # cancel very expensive only
    for s in subs:
        if s.cost > 1200:
            return Action(action_type="cancel", subscription_id=s.id)

    # otherwise keep
    return Action(
        action_type="keep",
        subscription_id=subs[0].id
    )


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

    # 🔥 NORMALIZE
    score = total_reward / MAX_TOTAL_REWARD

    # clamp to [0,1]
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
