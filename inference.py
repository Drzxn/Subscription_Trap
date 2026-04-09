import os
from openai import OpenAI

from core.env import SubscriptionEnv
from core.models import Action

# 🔹 Required environment variables
API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://dzrxn-subscription-trap.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ❗ DO NOT set default
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# 🔹 OpenAI client (required structure)
client = OpenAI(base_url=API_BASE_URL)


def smart_policy(obs):
    """
    Simple rule-based policy
    """

    # Cancel expensive subscriptions
    for sub in obs.visible_subscriptions:
        if sub.cost > obs.budget:
            return Action(
                action_type="cancel",
                subscription_id=sub.id
            )

    # Detect hidden trial from email
    for email in obs.email_logs:
        if "trial" in email["content"].lower():
            return Action(
                action_type="cancel",
                subscription_id="hidden_trial"
            )

    # Default safe action
    return Action(
        action_type="keep",
        subscription_id=obs.visible_subscriptions[0].id
        if obs.visible_subscriptions else "gym"
    )


def run_inference():
    print("STARTUP: Initializing agent")

    env = SubscriptionEnv()
    obs = env.reset()

    total_reward = 0.0
    step_count = 0

    while True:
        action = smart_policy(obs)

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)
        total_reward += reward_value
        step_count += 1

        print(
            f"STEP: step={step_count}, action={action.action_type}, "
            f"sub={action.subscription_id}, reward={reward_value}"
        )

        if done:
            break

    print(
        f"END: total_steps={step_count}, total_reward={round(total_reward, 2)}"
    )

    return total_reward


if __name__ == "__main__":
    run_inference()
