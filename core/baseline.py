from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 12.0  # 6 steps × max ~2 reward


def choose_action(obs, step):
    subs = obs.visible_subscriptions

    if not subs:
        return None

    # 🔥 Step-based strategy
    if step == 0:
        # investigate first (critical for hidden traps)
        target = subs[0]
        return Action(action_type="investigate", subscription_id=target.id)

    # prioritize risky subscriptions
    target = sorted(
        subs,
        key=lambda s: (s.hidden, s.trial, s.cost),
        reverse=True
    )[0]

    if target.hidden:
        return Action(action_type="investigate", subscription_id=target.id)

    if target.trial:
        return Action(action_type="cancel", subscription_id=target.id)

    if target.cost > 1000:
        return Action(action_type="cancel", subscription_id=target.id)

    return Action(action_type="keep", subscription_id=target.id)


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
