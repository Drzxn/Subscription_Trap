"""
grader.py

Evaluates agent actions and returns:
- reward (float)
- reason (string)

Design goals:
- deterministic
- stable reward range
- meaningful partial feedback
- avoids extreme penalty collapse
"""


def evaluate_step(state, truth, budget, action_count):
    reward = 0.0
    total_cost = 0.0

    active_subs = 0
    correct_actions = 0
    wrong_actions = 0
    reasons = []

    # 🔹 Evaluate each subscription
    for sub in state:
        correct = truth.get(sub.id)

        if correct is None:
            continue

        # 💰 Cost tracking
        if sub.active and not sub.trial:
            total_cost += sub.cost
            active_subs += 1

        # ✅ Correct cancel
        if not sub.active and correct == "cancel":
            reward += 0.6
            correct_actions += 1
            reasons.append(f"correctly cancelled {sub.id}")

        # ❌ Missed hidden trap (REDUCED PENALTY)
        if sub.hidden and sub.active and not sub.trial:
            reward -= 0.3   # ✅ FIXED (was too harsh before)
            wrong_actions += 1
            reasons.append(f"missed hidden trap {sub.id}")

        # ❌ Wrong cancel
        if not sub.active and correct == "keep":
            reward -= 0.4
            wrong_actions += 1
            reasons.append(f"wrongly cancelled {sub.id}")

    # 💰 Budget overflow penalty (controlled)
    if total_cost > budget:
        overflow = total_cost - budget
        penalty = min(overflow / 800.0, 0.5)  # softer scaling
        penalty = round(penalty, 2)

        reward -= penalty
        reasons.append(f"budget overflow penalty {penalty}")

    # ⚠️ Action spam penalty
    if action_count > 3:
        penalty = round((action_count - 3) * 0.1, 2)
        reward -= penalty
        reasons.append(f"action spam penalty {penalty}")

    # ⚠️ Too many active subscriptions
    if active_subs > 2:
        penalty = round(0.15 * active_subs, 2)
        reward -= penalty
        reasons.append(f"too many active subs penalty {penalty}")

    # 🎯 Efficiency bonus
    if correct_actions > 0 and wrong_actions == 0:
        reward += 0.3
        reasons.append("efficient decision bonus")

    # 🔥 Soft floor (prevents collapse)
    if reward < -2.0:
        reward = -2.0

    # 🔥 Final normalization
    reward = round(reward, 2)
    reward = max(min(reward, 2.0), -3.0)

    # 🧠 Clean reason output
    if not reasons:
        reason = "neutral step"
    else:
        reason = ", ".join(reasons[:3])

    return reward, reason
