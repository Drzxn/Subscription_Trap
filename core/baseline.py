from core.env import SubscriptionEnv
from core.models import Action


def run_baseline():
    env = SubscriptionEnv("hard")
    obs = env.reset()
    total_reward = 0

    for _ in range(6):
        for sub in obs.visible_subscriptions:
            # smarter baseline
            if sub.trial or sub.cost > 1000:
                action_type = "cancel"
            else:
                action_type = "keep"

            action = Action(action_type=action_type, subscription_id=sub.id)
            obs, reward, done, _ = env.step(action)
            total_reward += reward.value

            if done:
                break

    return total_reward
