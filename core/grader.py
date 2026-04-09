def evaluate_step(state, truth, budget, action_count):
    reward = 0.0
    total_cost = 0.0

    active_subs = 0
    correct_actions = 0
    wrong_actions = 0
    reasons = []

    for sub in state:
        correct = truth.get(sub.id)

        # Skip unknown safely
        if correct is None:
            continue

        # 💰 Cost calculation
        if sub.active and not sub.trial:
            total_cost += sub.cost
            active_subs += 1

        # ✅ Correct cancel
        if not sub.active and correct == "cancel":
            reward += 0.6
            correct_actions += 1
            reasons.append(f"correctly cancelled {sub.id}")

        # ❌ Missed hidden trap
        if sub.hidden and sub.active and not sub.trial:
            reward -= 1.0
            wrong_actions += 1
            reasons.append(f"missed hidden trap {sub.id}")

        # ❌ Unnecessary cancel
        if not sub.active and correct == "keep":
            reward -= 0.5
            wrong_actions += 1
            reasons.append(f"wrongly cancelled {sub.id}")

    # 💰 Budget overflow penalty (scaled)
    if total_cost > budget:
        overflow = total_cost - budget
        penalty = round(overflow / 100.0, 2)
        reward -= penalty
        reasons.append(f"budget overflow penalty {penalty}")

    # ⚠️ Action spam penalty
    if action_count > 3:
        penalty = round((action_count - 3) * 0.1, 2)
        reward -= penalty
        reasons.append(f"action spam penalty {penalty}")

    # ⚠️ Too many active subscriptions
    if active_subs > 2:
        penalty = round(0.2 * active_subs, 2)
        reward -= penalty
        reasons.append(f"too many active subs penalty {penalty}")

    # 🎯 Efficiency bonus
    if correct_actions > 0 and wrong_actions == 0:
        reward += 0.3
        reasons.append("efficient decision bonus")

    # 🔥 FINAL CLEANUP (CRITICAL)
    reward = round(reward, 2)
    reward = max(min(reward, 2.0), -3.0)

    # 🧠 Better explanation
    reason = ", ".join(reasons) if reasons else "neutral step"

    return reward, reason
