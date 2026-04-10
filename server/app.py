from fastapi import FastAPI, HTTPException
from typing import Any

from core.env import SubscriptionEnv
from core.baseline import run_baseline
from core.models import Action

app = FastAPI(
    title="Subscription Trap OpenEnv",
    version="1.0.0"
)

# 🔥 GLOBAL ENV (default task)
env = SubscriptionEnv("easy")


# 🔹 Safe serializer (handles Pydantic)
def serialize(obj: Any):
    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if isinstance(obj, list):
        return [serialize(x) for x in obj]

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    return obj


# 🔹 HEALTH
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 ROOT
@app.get("/")
def root():
    return {"message": "Subscription Trap Running"}


# 🔥 RESET (CRITICAL FIX FOR VALIDATOR)
@app.api_route("/reset", methods=["GET", "POST"])
def reset(task: str = "easy"):
    try:
        global env

        # ✅ MULTI-TASK SWITCHING (VERY IMPORTANT)
        env = SubscriptionEnv(task)

        obs = env.reset()

        return {
            "observation": serialize(obs),
            "reward": {"value": 0.0, "reason": "reset"},
            "done": False,
            "info": {
                "task": task,
                "month": 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 STEP
@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)

        return {
            "observation": serialize(obs),
            "reward": {
                "value": float(getattr(reward, "value", 0.0)),
                "reason": getattr(reward, "reason", "")
            },
            "done": bool(done),
            "info": serialize(info)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 BASELINE
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


# 🔹 GRADER SNAPSHOT
@app.get("/grader")
def grader():
    try:
        subs = env.state or []

        active = [s.id for s in subs if getattr(s, "active", False)]
        total = len(subs) if subs else 1

        return {
            "active_subscriptions": active,
            "total_active": len(active),
            "score_hint": round(1.0 - (len(active) / total), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 TASKS (CRITICAL FOR VALIDATOR)
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "difficulty": 1},
            {"name": "medium", "difficulty": 2},
            {"name": "hard", "difficulty": 3}
        ],
        "action_schema": {
            "action_type": ["cancel", "keep", "snooze", "investigate"],
            "subscription_id": "string"
        }
    }


# 🔥 REQUIRED FOR DOCKER VALIDATOR
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
