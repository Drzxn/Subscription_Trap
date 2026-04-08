from fastapi import FastAPI
from env import SubscriptionEnv
from baseline import run_baseline
from models import Action

app = FastAPI()

env = SubscriptionEnv()


# 🔹 Reset environment
@app.get("/reset")
def reset():
    obs = env.reset()
    return {
        "observation": obs
    }


# 🔹 Step function (FIXED)
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": {
            "value": reward.value,
            "reason": reward.reason
        },
        "done": done,
        "info": info
    }


# 🔹 Baseline (reproducible)
@app.get("/baseline")
def baseline():
    score = run_baseline()
    return {
        "baseline_score": score,
        "description": "Rule-based baseline agent"
    }


# 🔹 Grader (improved)
@app.get("/grader")
def grader():
    state = env.state

    return {
        "active_subscriptions": [
            s.id for s in state if s.active
        ],
        "total_active": len([s for s in state if s.active])
    }


# 🔹 Tasks (already good, kept but structured)
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Basic subscriptions with no hidden traps",
            },
            {
                "name": "medium",
                "description": "Trial subscriptions with delayed cost",
            },
            {
                "name": "hard",
                "description": "Hidden subscriptions and misleading signals",
            }
        ],
        "action_schema": {
            "action_type": ["cancel", "keep", "snooze", "investigate"],
            "subscription_id": "string"
        }
    }
