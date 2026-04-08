from pydantic import BaseModel
from typing import List


class Subscription(BaseModel):
    id: str
    cost: float
    active: bool
    trial: bool
    trial_remaining: int
    hidden: bool


class BankLog(BaseModel):
    description: str
    amount: float


class EmailLog(BaseModel):
    subject: str
    content: str


class Observation(BaseModel):
    visible_subscriptions: List[Subscription]
    bank_logs: List[BankLog]
    email_logs: List[EmailLog]
    month: int
    budget: float
    action_count: int


class Action(BaseModel):
    action_type: str  # cancel / keep / snooze / investigate
    subscription_id: str


class Reward(BaseModel):
    value: float
    reason: str
