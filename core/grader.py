def evaluate_step(state, truth, budget, action_count):
    reward = 0.0
    total_cost = 0.0

    active_subs = 0
    correct_actions = 0
    wrong_actions = 0

    for sub in state:
        # Safe lookup (prevents crash)
        correct = truth.get(sub.id)

        # Skip if unknown (safety)
        if correct is None:
            continue

        # Calculate cost
        if sub.active and not sub.trial:
            total_cost += sub.cost
            active_subs += 1

        # ✅ Correct decision
        if not sub.active and correct == "cancel":
            reward += 0.6
            correct_actions += 1

        # ❌ Missed hidden trap (VERY IMPORTANT)
        if sub.hidden and sub.active and not sub.trial:
            reward -= 1.2
            wrong_actions += 1

        # ❌ Unnecessary cancel
        if not sub.active and correct == "keep":
            reward -= 0.6
            wrong_actions += 1

    # 💰 Budget penalty (scaled)
    if total_cost > budget:
        overflow = total_cost - budget
        reward -= overflow / 100.0

    # ⚠️ Action spam penalty (smarter)
    if action_count > 3:
        reward -= (action_count - 3) * 0.1

    # ⚠️ Too many active subscriptions penalty
    if active_subs > 2:
        reward -= 0.2 * active_subs

    # 🎯 Small bonus for efficiency
    if correct_actions > 0 and wrong_actions == 0:
        reward += 0.3

    # 🧱 Clamp reward (important for stability)
    reward = max(min(reward, 2.0), -3.0)

    return reward, "final_evaluation"
