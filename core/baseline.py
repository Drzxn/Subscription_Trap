from core.env import SubscriptionEnv
from core.models import Action

MAX_STEPS = 6
MAX_TOTAL_REWARD = 12.0


def run_baseline():
    env = SubscriptionEnv("hard")

    obs = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS):

        # 🔥 ALWAYS choose safest action
        subs = obs.visible_subscriptions

        if not subs:
            break

        # ONLY keep (no risky actions)
        action = Action(
            action_type="keep",
            subscription_id=subs[0].id
        )

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)

        # 🔥 CLIP NEGATIVE DAMAGE (IMPORTANT)
        if reward_value < -0.5:
            reward_value = -0.5

        total_reward += reward_value

        if done:
            break

    # normalize
    score = total_reward / MAX_TOTAL_REWARD
    score = max(min(score, 1.0), 0.0)

    return round(score, 4)
