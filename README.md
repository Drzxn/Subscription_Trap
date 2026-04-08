---
title: Subscription Trap Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Subscription Trap Escape Environment

## Overview
Simulates subscription management with hidden costs.



## Observation Space
- visible_subscriptions
- bank_logs
- email_logs
- month
- budget

## Action Space
- cancel
- keep
- snooze

## Tasks
- Easy: Cancel visible subscriptions
- Medium: Detect hidden subscriptions
- Hard: Optimize cost over time

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload