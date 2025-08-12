"""Shim module exposing QATestAgent under dev_guard.agents.qa_test for test patches.

Some tests patch dev_guard.agents.qa_test.QATestAgent. Our implementation
lives in qa_agent.py, so we re-export the symbol here for compatibility.
"""

from .qa_agent import QATestAgent

