# tests/security/test_z6_t2e_dev_gate.py
"""Z6 T2E — dev vault fallback is gated behind explicit env var.

Without `KUTAY_MASTER_KEY` AND without `KUTAY_DEV_ALLOW_INSECURE_VAULT=1`,
`_get_fernet()` (and therefore `_encrypt`/`_decrypt`) must raise
RuntimeError. With the opt-in env, the base64 path still works but every
call emits a warning containing hostname + pid for grep-ability.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class _BaseDevGate(unittest.TestCase):
    def setUp(self):
        try:
            from cryptography.fernet import Fernet  # noqa: F401
        except ImportError:
            self.skipTest("cryptography not installed")

        import src.security.credential_store as cs_mod
        self.cs_mod = cs_mod
        cs_mod._reset_key_state()
        self._env_backup = {
            k: os.environ.get(k)
            for k in (
                "KUTAY_MASTER_KEY",
                "KUTAY_MASTER_KEY_v1",
                "KUTAY_MASTER_KEY_v2",
                "KUTAY_DEV_ALLOW_INSECURE_VAULT",
                "KUTAY_ENV",
            )
        }
        for k in self._env_backup:
            os.environ.pop(k, None)

    def tearDown(self):
        self.cs_mod._reset_key_state()
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestNoEnvRaises(_BaseDevGate):
    def test_no_key_no_dev_flag_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            self.cs_mod._get_fernet()
        msg = str(cm.exception)
        self.assertIn("KUTAY_DEV_ALLOW_INSECURE_VAULT", msg)

    def test_encrypt_raises_too(self):
        with self.assertRaises(RuntimeError):
            self.cs_mod._encrypt("hello")


class TestOptInWorks(_BaseDevGate):
    def test_dev_flag_enables_base64_fallback(self):
        os.environ["KUTAY_DEV_ALLOW_INSECURE_VAULT"] = "1"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enc = self.cs_mod._encrypt("payload")
            dec = self.cs_mod._decrypt(enc)
        self.assertEqual(dec, "payload")
        msgs = [str(x.message) for x in w]
        # Warning for encrypt AND decrypt — every call.
        self.assertTrue(any("insecure base64 fallback" in m for m in msgs))

    def test_warnings_every_call_not_once(self):
        import socket

        os.environ["KUTAY_DEV_ALLOW_INSECURE_VAULT"] = "1"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(3):
                self.cs_mod._encrypt("payload")
        msgs = [str(x.message) for x in w]
        insecure = [m for m in msgs if "insecure base64 fallback" in m]
        self.assertGreaterEqual(len(insecure), 3)
        # Hostname + pid present for grep-ability
        host = socket.gethostname()
        self.assertTrue(any(host in m for m in insecure))
        self.assertTrue(any(str(os.getpid()) in m for m in insecure))


class TestRealKeyBypassesGate(_BaseDevGate):
    def test_master_key_set_no_warning(self):
        os.environ["KUTAY_MASTER_KEY"] = "real-master-key-bypass-test"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enc = self.cs_mod._encrypt("payload")
            dec = self.cs_mod._decrypt(enc)
        self.assertEqual(dec, "payload")
        # No insecure-fallback warning should fire.
        msgs = [str(x.message) for x in w]
        self.assertFalse(
            any("insecure base64 fallback" in m for m in msgs),
            msgs,
        )


if __name__ == "__main__":
    unittest.main()
