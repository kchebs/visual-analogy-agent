"""Compatibility entrypoint — prefer ``from rpm_agent import Agent``."""
from rpm_agent.agent import Agent

__all__ = ["Agent"]
