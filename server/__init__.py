# server/__init__.py

"""
Server package for Subscription Trap OpenEnv.

Exposes FastAPI app and entrypoint.
"""

from .app import app, main

__all__ = [
    "app",
    "main",
]
