"""Backward-compatible wrapper for problem formulation reporting.

Use `src.report_problem` as the canonical implementation.
"""

from .report_problem import build_problem_formulation_markdown

__all__ = ["build_problem_formulation_markdown"]
