import unittest
from io import BytesIO
from urllib.error import HTTPError

from rag_eval_pipeline.http import describe_http_error, payload_summary, response_preview, response_summary


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

    def test_response_preview_flattens_and_truncates_body(self):
        preview = response_preview("line1\nline2", limit=8)

        self.assertEqual(preview, "line1\\nl...<truncated>")

    def test_response_summary_reports_collection_shapes(self):
        summary = response_summary({"code": 0, "data": {"chunks": []}, "message": "success"})

        self.assertIn("code=0", summary)
        self.assertIn("data=dict(keys=['chunks'])", summary)
        self.assertIn("message='success'", summary)


if __name__ == "__main__":
    unittest.main()
