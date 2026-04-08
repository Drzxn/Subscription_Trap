from models import Subscription, BankLog, EmailLog


def get_task(level):
    subs = [
        Subscription(id="gym", cost=1500, active=True,
                     trial=False, trial_remaining=0, hidden=False),
        Subscription(id="hidden_trial", cost=999, active=True,
                     trial=True, trial_remaining=1, hidden=True),
    ]

    truth = {
        "gym": "keep",
        "hidden_trial": "cancel"
    }

    bank_logs = [
        BankLog(description="GYM MEMBERSHIP", amount=-1500),
        BankLog(description="FREE TRIAL SERVICE", amount=0),
        BankLog(description="UNKNOWN APP STORE", amount=-999),
    ]

    email_logs = [
        EmailLog(subject="Welcome to Premium!",
                 content="Your trial ends soon."),
        EmailLog(subject="Special Offer", content="Upgrade now!"),
    ]

    return {
        "subscriptions": subs,
        "ground_truth": truth,
        "bank_logs": bank_logs,
        "email_logs": email_logs,
        "budget": 1200
    }
