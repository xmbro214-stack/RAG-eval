import unittest
from io import BytesIO
from urllib.error import HTTPError

from rag_eval_pipeline.http import describe_http_error, payload_summary


class TestHttpHelpers(unittest.TestCase):
    def test_payload_summary_redacts_and_summarizes_messages(self):
        summary = payload_summary({
            "model": "demo",
            "Authorization": "Bearer secret",
            "messages": [{"role": "user", "content": "hello"}],
        })

        self.assertIn("Authorization=<redacted>", summary)
        self.assertIn("content_chars", summary)
        self.assertNotIn("secret", summary)

    def test_describe_http_error_includes_response_body(self):
        error = HTTPError(
            url="http://example.test",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"error":"bad"}'),
        )

        self.assertIn('{"error":"bad"}', describe_http_error(error))


if __name__ == "__main__":
    unittest.main()
