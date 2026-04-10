import os
from openai import OpenAI

from core.env import SubscriptionEnv
from core.models import Action

# 🔥 REQUIRED FOR VALIDATOR (LLM PROXY)
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL"),
    api_key=os.environ.get("API_KEY")
)

TASKS = ["easy", "medium", "hard"]


def choose_action(obs):
    # simple safe policy
    for sub in obs.visible_subscriptions:
        if sub.cost > obs.budget:
            return Action(action_type="cancel", subscription_id=sub.id)

    return Action(
        action_type="keep",
        subscription_id=obs.visible_subscriptions[0].id
        if obs.visible_subscriptions else "gym"
    )


def run_task(task_name):
    print(f"[START] task={task_name}", flush=True)

    env = SubscriptionEnv(task_name)
    obs = env.reset()

    total_reward = 0.0
    step = 0

    while True:
        action = choose_action(obs)

        obs, reward, done, _ = env.step(action)

        reward_value = getattr(reward, "value", 0.0)
        total_reward += reward_value
        step += 1

        print(
            f"[STEP] task={task_name} step={step} reward={reward_value}",
            flush=True
        )

        if done:
            break

    score = round(total_reward, 2)

    print(
        f"[END] task={task_name} score={score} steps={step}",
        flush=True
    )

    return score


def run_inference():
    results = {}

    for task in TASKS:
        try:
            score = run_task(task)
            results[task] = score
        except Exception as e:
            print(f"[ERROR] task={task} error={str(e)}", flush=True)
            results[task] = 0.0

    return results


if __name__ == "__main__":
    run_inference()
