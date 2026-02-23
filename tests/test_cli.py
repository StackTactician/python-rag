import io
import sys
import types
import unittest
from unittest.mock import patch

# Stub optional deps needed by src.rag import chain.
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
sys.modules.setdefault("requests", types.SimpleNamespace(RequestException=Exception, post=lambda **kwargs: None))

from src import cli


class CliTest(unittest.TestCase):
    def test_single_query_streams_output(self):
        fake_stream = [("Hello", []), (" world", [])]

        with patch.object(cli, "query_rag", return_value=iter(fake_stream)):
            output = io.StringIO()
            with patch("sys.stdout", output):
                cli.single_query("hi", model_name="model-a")

        text = output.getvalue()
        self.assertIn("Using Model: model-a", text)
        self.assertIn("Hello world", text)


if __name__ == "__main__":
    unittest.main()
