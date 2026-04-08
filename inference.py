from core.env import SubscriptionEnv


def smart_policy(obs):
    """
    Intelligent decision-making policy
    """

    # 🔍 Look for hidden or suspicious subscriptions
    for sub in obs.visible_subscriptions:
        # Cancel expensive subscriptions over budget
        if sub.cost > obs.budget:
            return {
                "action_type": "cancel",
                "subscription_id": sub.id
            }

    # 🔍 Look for trial traps via email clues
    for email in obs.email_logs:
        if "trial ends soon" in email["content"].lower():
            # Try to cancel hidden trial
            return {
                "action_type": "cancel",
                "subscription_id": "hidden_trial"
            }

    # Default: do nothing harmful
    return {
        "action_type": "keep",
        "subscription_id": "gym"
    }


def run_inference():
    env = SubscriptionEnv()

    obs = env.reset()
    total_reward = 0
    steps = 0

    print("\n🚀 Starting Inference Run\n")

    while True:
        action = smart_policy(obs)

        obs, reward, done, _ = env.step(type("A", (), action))

        total_reward += reward.value
        steps += 1

        print(f"Step {steps}")
        print(f"Action: {action}")
        print(f"Reward: {reward.value} | Reason: {reward.reason}")
        print(f"Month: {obs.month}")
        print("-" * 40)

        if done:
            break

    print("\n✅ Inference Completed")
    print(f"Total Steps: {steps}")
    print(f"Final Score: {total_reward}\n")

    return total_reward


if __name__ == "__main__":
    run_inference()
