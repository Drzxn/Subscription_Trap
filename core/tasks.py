from typing import Dict, List
import random

from core.models import Subscription, BankLog, EmailLog


def get_task(level: str = "hard") -> Dict:
    """
    Generate task scenarios with increasing difficulty.
    """

    # 🟢 EASY — no hidden traps
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

    # 🟡 MEDIUM — trial-based traps
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

    # 🔴 HARD — hidden + misleading signals
    else:
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

    # 🔄 Add slight randomness (important for RL)
    random.shuffle(subs)

    # 💳 Bank logs (derived from subscriptions)
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

    # 📧 Email logs (weak signals / distractions)
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

    # 🎯 Budget constraint (forces decisions)
    budget = sum([s.cost for s in subs]) * 0.6

    return {
        "subscriptions": subs,
        "ground_truth": truth,
        "bank_logs": bank_logs,
        "email_logs": email_logs,
        "budget": budget
    }
