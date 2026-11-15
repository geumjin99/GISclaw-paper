"""Locations inside the repository. Every script takes its paths from this module.

The input data of GeoAnalystBench and the reference outputs built from them are not
part of the repository; their locations can be set with environment variables.
"""
import os

ROOT = os.environ.get("GISCLAW_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AGENT_DIR = os.path.join(ROOT, "agent")
HARNESS_DIR = os.path.join(ROOT, "harness")

BENCH = os.path.join(ROOT, "benchmark")
TASKS_CSV = os.path.join(BENCH, "tasks.csv")
SPECS = os.path.join(BENCH, "specs")
NOTES = os.path.join(BENCH, "knowledge_notes")
PERTURBATIONS = os.path.join(BENCH, "perturbations")

INPUTS = os.environ.get("GISCLAW_INPUTS") or os.path.join(BENCH, "inputs")
GOLD = os.environ.get("GISCLAW_GOLD") or os.path.join(BENCH, "gold")
RESULTS = os.environ.get("GISCLAW_RESULTS") or os.path.join(ROOT, "results")
