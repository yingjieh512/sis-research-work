"""Small subset of absl.testing.absltest backed by unittest."""

from __future__ import annotations

import unittest


class TestCase(unittest.TestCase):
  """Compatibility TestCase with the assertLen helper used by SIS tests."""

  def assertLen(self, container, expected_len):
    self.assertEqual(len(container), expected_len)


def main(*args, **kwargs):
  return unittest.main(*args, **kwargs)
