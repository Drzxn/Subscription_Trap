from typing import Dict, List

from core.models import Subscription, BankLog, EmailLog


def get_task(level: str = "hard") -> Dict:
    """
    Deterministic task generator (REQUIRED for OpenEnv validation)
    """

    # =========================
    # 🟢 EASY TASK
    # =========================
    if level == "easy":
        subs = [
            Subscription(
                id="netflix",
                cost=499,
                active=True,
                trial=False,
                trial_remaining=0,
                hidden=False
            ),
            Subscription(
                id="spotify",
                cost=199,
                active=True,
                trial=False,
                trial_remaining=0,
                hidden=False
            ),
        ]

        truth = {
            "netflix": "keep",
            "spotify": "keep"
        }

    # =========================
    # 🟡 MEDIUM TASK
    # =========================
    elif level == "medium":
        subs = [
            Subscription(
                id="gym",
                cost=1500,
                active=True,
                trial=False,
                trial_remaining=0,
                hidden=False
            ),
            Subscription(
                id="trial_app",
                cost=799,
                active=True,
                trial=True,
                trial_remaining=1,
                hidden=False
            ),
        ]

        truth = {
            "gym": "keep",
            "trial_app": "cancel"
        }

    # =========================
    # 🔴 HARD TASK
    # =========================
    elif level == "hard":
        subs = [
            Subscription(
                id="gym",
                cost=1500,
                active=True,
                trial=False,
                trial_remaining=0,
                hidden=False
            ),
            Subscription(
                id="hidden_trial",
                cost=999,
                active=True,
                trial=True,
                trial_remaining=1,
                hidden=True
            ),
            Subscription(
                id="fake_free",
                cost=0,
                active=True,
                trial=False,
                trial_remaining=0,
                hidden=False
            ),
        ]

        truth = {
            "gym": "keep",
            "hidden_trial": "cancel",
            "fake_free": "keep"
        }

    else:
        # 🔥 HARD fallback (REQUIRED)
        return get_task("hard")

    # =========================
    # 💳 BANK LOGS
    # =========================
    bank_logs: List[BankLog] = []
    for sub in subs:
        if sub.cost > 0:
            bank_logs.append(
                BankLog(
                    description=f"{sub.id.upper()} PAYMENT",
                    amount=-sub.cost
                )
            )
        else:
            bank_logs.append(
                BankLog(
                    description=f"{sub.id.upper()} FREE",
                    amount=0
                )
            )

    # =========================
    # 📧 EMAIL LOGS
    # =========================
    email_logs: List[EmailLog] = [
        EmailLog(
            subject="Welcome!",
            content="Enjoy your subscription."
        ),
        EmailLog(
            subject="Limited Offer",
            content="Upgrade to premium now!"
        ),
        EmailLog(
            subject="Trial Ending Soon",
            content="Your trial expires in 24 hours."
        ),
    ]

    # =========================
    # 🎯 BUDGET (deterministic)
    # =========================
    total_cost = sum([s.cost for s in subs])
    budget = round(total_cost * 0.6, 2)

    return {
        "subscriptions": subs,
        "ground_truth": truth,
        "bank_logs": bank_logs,
        "email_logs": email_logs,
        "budget": budget
    }
