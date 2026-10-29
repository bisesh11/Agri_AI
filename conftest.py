"""Pytest configuration to ensure workspace root is on sys.path."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

