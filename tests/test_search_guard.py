"""Tests for the search-required guard in BaseAgent."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.base import BaseAgent


class TestGetSearchDepth(unittest.TestCase):

    def test_extracts_from_classification_context(self):
        task = {"context": json.dumps({"classification": {"search_depth": "deep"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "deep")

    def test_extracts_quick(self):
        task = {"context": json.dumps({"classification": {"search_depth": "quick"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "quick")

    def test_extracts_standard(self):
        task = {"context": json.dumps({"classification": {"search_depth": "standard"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "standard")

    def test_returns_none_for_no_search(self):
        task = {"context": json.dumps({"classification": {"search_depth": "none"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_no_context(self):
        task = {}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_context_is_empty_string(self):
        task = {"context": ""}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_context_is_invalid_json(self):
        task = {"context": "not json"}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_classification_missing(self):
        task = {"context": json.dumps({"other": "data"})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_returns_none_when_search_depth_missing(self):
        task = {"context": json.dumps({"classification": {"agent_type": "coder"}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")

    def test_handles_dict_context(self):
        task = {"context": {"classification": {"search_depth": "deep"}}}
        self.assertEqual(BaseAgent._get_search_depth(task), "deep")

    def test_handles_none_search_depth_value(self):
        task = {"context": json.dumps({"classification": {"search_depth": None}})}
        self.assertEqual(BaseAgent._get_search_depth(task), "none")


if __name__ == "__main__":
    unittest.main()
