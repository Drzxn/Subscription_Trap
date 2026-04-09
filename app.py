from fastapi import FastAPI, HTTPException
from typing import Any

from core.env import SubscriptionEnv
from core.baseline import run_baseline
from core.models import Action

app = FastAPI(
    title="Subscription Trap OpenEnv",
    version="1.0.0",
    description="RL environment for detecting hidden subscription traps"
)

# Initialize environment
env = SubscriptionEnv()


# 🔹 Safe serialization
def serialize(obj: Any):
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if isinstance(obj, (list, tuple)):
        return [serialize(o) for o in obj]

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    return obj


# 🔹 Health check (HF uses this sometimes)
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 Root
@app.get("/")
def root():
    return {
        "message": "Subscription Trap Environment Running",
        "version": "1.0.0"
    }


# 🔹 Reset
@app.get("/reset")
def reset():
    try:
        obs = env.reset()
        return {
            "observation": serialize(obs),
            "reward": {"value": 0.0, "reason": "reset"},
            "done": False,
            "info": {
                "message": "environment reset",
                "month": 0,
                "note": "some subscriptions may be hidden"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 Step
@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)

        return {
            "observation": serialize(obs),
            "reward": {
                "value": float(getattr(reward, "value", reward or 0.0)),
                "reason": getattr(reward, "reason", "")
            },
            "done": bool(done),
            "info": serialize(info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 State
@app.get("/state")
def state():
    try:
        return {"state": serialize(env.state())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 Baseline
@app.get("/baseline")
def baseline():
    try:
        score = run_baseline()
        return {
            "baseline_score": float(score),
            "range": "0.0 - 1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 Grader
@app.get("/grader")
def grader():
    try:
        state = env.state()
        subs = state.get("subscriptions", [])

        active = [s["id"] for s in subs if s.get("active")]

        total = len(subs) if subs else 1

        return {
            "active_subscriptions": active,
            "total_active": len(active),
            "score_hint": round(1.0 - (len(active) / total), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 Tasks
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "difficulty": 1},
            {"name": "medium", "difficulty": 2},
            {"name": "hard", "difficulty": 3}
        ],
        "action_schema": {
            "action_type": ["cancel", "keep", "snooze"],
            "subscription_id": "string"
        }
    }
