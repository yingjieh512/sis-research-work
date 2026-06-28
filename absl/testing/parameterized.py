"""Small subset of absl.testing.parameterized backed by unittest."""

from __future__ import annotations

import functools

from absl.testing import absltest


class TestCase(absltest.TestCase):
  pass


def named_parameters(*testcases):
  """Expands absl-style named parameter dictionaries into subTests."""

  def decorator(test_method):

    @functools.wraps(test_method)
    def wrapper(self):
      for testcase in testcases:
        params = dict(testcase)
        testcase_name = params.pop("testcase_name", str(params))
        with self.subTest(testcase_name):
          test_method(self, **params)

    return wrapper

  return decorator
