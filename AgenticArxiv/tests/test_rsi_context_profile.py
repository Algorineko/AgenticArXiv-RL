import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rsi_context_profile', Path(__file__).resolve().parents[2] / 'scripts/rsi_qwen_context_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

class ContextGuardTests(unittest.TestCase):
    def test_guards_global_and_allocator_headroom(self):
        self.assertTrue(profile.allowed(16000, 18000, 46000, 16000)[0])
        self.assertFalse(profile.allowed(16000, 18000, 46000, 12000)[0])
        self.assertFalse(profile.allowed(16000, 22000, 46000, 16000)[0])
