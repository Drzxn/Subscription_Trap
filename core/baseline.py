from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 12.0


def choose_action(obs, step):
    subs = obs.visible_subscriptions

    if not subs:
        return None

    # 🔥 STEP 0–1: exploration phase
    if step < 2:
        for s in subs:
            if getattr(s, "hidden", False):
                return Action(action_type="investigate", subscription_id=s.id)

        # fallback: investigate first visible
        return Action(action_type="investigate", subscription_id=subs[0].id)

    # 🔥 AFTER exploration: decision phase

    # prioritize dangerous subs
    target = sorted(
        subs,
        key=lambda s: (getattr(s, "hidden", False), s.trial, s.cost),
        reverse=True
    )[0]

    if getattr(target, "hidden", False):
        return Action(action_type="investigate", subscription_id=target.id)

    if target.trial:
        return Action(action_type="cancel", subscription_id=target.id)

    if target.cost > 800:   # 🔥 tuned threshold
        return Action(action_type="cancel", subscription_id=target.id)

    return Action(action_type="keep", subscription_id=target.id)


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

    # 🔥 normalize
    score = total_reward / MAX_TOTAL_REWARD
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
