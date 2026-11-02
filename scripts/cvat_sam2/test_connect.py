"""Offline launcher checks; no CVAT calls, credentials or annotation writes."""
import unittest
from unittest.mock import patch

import connect


class TestFunctionReuse(unittest.TestCase):
    def lookup(self, entries):
        with patch.object(connect, "api_get", return_value={"results": entries}) as api:
            result = connect.find_function("dummy-not-a-credential", "test-user")
            self.assertIn("/api/functions?", api.call_args.args[0])
            return result

    def test_new(self):
        self.assertIsNone(self.lookup([]))

    def test_private_tracker_reuse(self):
        self.assertEqual(self.lookup([{
            "id": 123, "visibility": "private", "kind": "tracker", "provider": "native",
        }]), 123)

    def test_no_public_function_reuse(self):
        with self.assertRaises(RuntimeError):
            self.lookup([{"id": 123, "visibility": "public", "kind": "tracker", "provider": "native"}])

    def test_no_detector_reuse(self):
        with self.assertRaises(RuntimeError):
            self.lookup([{"id": 123, "visibility": "private", "kind": "detector", "provider": "native"}])

    def test_no_ambiguous_reuse(self):
        with self.assertRaises(RuntimeError):
            self.lookup([{"id": 1}, {"id": 2}])

    def test_no_non_native_reuse(self):
        with self.assertRaises(RuntimeError):
            self.lookup([{"id": 123, "visibility": "private", "kind": "tracker", "provider": "nuclio"}])


if __name__ == "__main__":
    unittest.main()
