from fastapi import FastAPI, HTTPException
from typing import Any
import uvicorn

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


# 🔹 Safe serialization (CRITICAL)
def serialize(obj: Any):
    try:
        if obj is None:
            return None

        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()

        if hasattr(obj, "dict"):  # fallback
            return obj.dict()

        if isinstance(obj, (list, tuple)):
            return [serialize(o) for o in obj]

        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}

        return obj

    except Exception:
        return str(obj)


# 🔹 Health check (MANDATORY for HF)
@app.get("/health")
def health():
    return {"status": "ok"}


# 🔹 Root endpoint
@app.get("/")
def root():
    return {
        "message": "Subscription Trap Environment Running",
        "version": "1.0.0"
    }


# 🔹 Reset environment
@app.get("/reset")
def reset():
    try:
        obs = env.reset()

        return {
            "observation": serialize(obs),
            "reward": {"value": 0.0, "reason": "reset"},
            "done": False,
            "info": {"message": "Environment reset successful"}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# 🔹 Step function (STRICT CONTRACT)
@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)

        # Safe reward handling
        reward_value = 0.0
        reward_reason = "N/A"

        if reward is not None:
            if hasattr(reward, "value"):
                reward_value = float(reward.value)
                reward_reason = getattr(reward, "reason", "N/A")
            else:
                reward_value = float(reward)

        return {
            "observation": serialize(obs),
            "reward": {
                "value": reward_value,
                "reason": reward_reason
            },
            "done": bool(done),
            "info": serialize(info)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# 🔹 State endpoint
@app.get("/state")
def state():
    try:
        return {
            "state": serialize(env.state)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State failed: {str(e)}")


# 🔹 Baseline evaluation
@app.get("/baseline")
def baseline():
    try:
        score = run_baseline()

        return {
            "baseline_score": float(score),
            "range": "0.0 - 1.0",
            "description": "Rule-based baseline agent"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Baseline failed: {str(e)}")


# 🔹 Grader snapshot
@app.get("/grader")
def grader():
    try:
        state = env.state or []

        active = [s.id for s in state if getattr(s, "active", False)]

        total = len(state) if len(state) > 0 else 1

        return {
            "active_subscriptions": active,
            "total_active": len(active),
            "score_hint": round(1.0 - (len(active) / total), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grader failed: {str(e)}")


# 🔹 Task definitions
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Basic subscriptions with no hidden traps",
                "difficulty": 1
            },
            {
                "name": "medium",
                "description": "Trial subscriptions with delayed cost",
                "difficulty": 2
            },
            {
                "name": "hard",
                "description": "Hidden subscriptions and misleading signals",
                "difficulty": 3
            }
        ],
        "action_schema": {
            "action_type": ["cancel", "keep", "snooze", "investigate"],
            "subscription_id": "string"
        }
    }


# 🔥 ENTRYPOINT (MANDATORY for Docker/OpenEnv)
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# 🔥 Required for direct execution
if __name__ == "__main__":
    main()
