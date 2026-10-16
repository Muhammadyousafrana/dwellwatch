"""
Smoke + unit tests for dwell-monitor.

Run (mocked only, no live server needed):
    pytest

Run with live local inference server:
    set ROBOFLOW_API_KEY=<key> && set SAMPLE_IMAGE=path\\to\\frame.jpg && pytest
"""
import base64
import os
from pathlib import Path

import pytest

import workflow_client as wc

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FAKE = {
    "outputs": [
        {
            "annotated_image": {"type": "base64", "value": "QUJD"},
            "violations": 1,
            "alert_status": "SECURITY ALERT",
            "total_vehicles": 1,
            "predictions": {
                "predictions": [{"class": "car", "tracker_id": 3, "time_in_zone": 6.2}]
            },
            "slack_notification_status": False,
            "slack_notification_message": "sent",
        }
    ]
}


class MockResp:
    """Minimal mock of requests.Response."""
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self.text = str(data or "")
        self._data = data if data is not None else FAKE

    def json(self):
        return self._data


# ---------------------------------------------------------------------------
# workflow_client tests
# ---------------------------------------------------------------------------

class TestRunDwellWorkflow:
    def test_parsing_with_mock(self, monkeypatch):
        """Happy path: correct request structure and response parsing."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
        sent = {}

        def post(url, json, headers, timeout):
            sent.update(url=url, json=json, headers=headers)
            return MockResp()

        monkeypatch.setattr(wc.requests, "post", post)
        entry = wc.run_dwell_workflow("QUJD")

        # Response keys present
        assert wc.EXPECTED_KEYS <= set(entry)

        # BUG FIX check: api_key is now in the body, not the Authorization header
        assert sent["json"]["api_key"] == "test-key"
        assert "Authorization" not in sent["headers"]

        # BUG FIX check: video_metadata is top-level, not nested inside image
        inputs = sent["json"]["inputs"]
        assert "video_metadata" in inputs, "video_metadata must be a top-level input"
        assert "video_metadata" not in inputs.get("image", {}), \
            "video_metadata must NOT be nested inside the image dict"

        # Summarise correctly
        summary = wc.summarize(entry)
        assert summary["vehicles"][0]["time_in_zone"] == 6.2
        assert summary["annotated_image"] == "QUJD"

    def test_missing_api_key(self, monkeypatch):
        """Missing API key raises WorkflowAuthError before any network call."""
        monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
        with pytest.raises(wc.WorkflowAuthError, match="ROBOFLOW_API_KEY"):
            wc.run_dwell_workflow("QUJD")

    def test_auth_rejected_401(self, monkeypatch):
        """401 from server raises WorkflowAuthError immediately (no retry)."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "bad-key")
        calls = [0]

        def post(*a, **kw):
            calls[0] += 1
            return MockResp(status_code=401)

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowAuthError):
            wc.run_dwell_workflow("QUJD")
        assert calls[0] == 1, "should NOT retry on auth error"

    def test_server_error_retried_then_raised(self, monkeypatch):
        """5xx responses are retried; WorkflowHTTPError raised after exhaustion."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
        monkeypatch.setattr(wc.time, "sleep", lambda _: None)  # skip back-off
        calls = [0]

        def post(*a, **kw):
            calls[0] += 1
            return MockResp(status_code=503)

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowHTTPError, match="server error 503"):
            wc.run_dwell_workflow("QUJD", retries=2)
        assert calls[0] == 3, "initial attempt + 2 retries = 3 calls"

    def test_timeout_retried_then_raised(self, monkeypatch):
        """Timeouts are retried; WorkflowTimeoutError raised after exhaustion."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
        monkeypatch.setattr(wc.time, "sleep", lambda _: None)
        import requests as req_lib

        def post(*a, **kw):
            raise req_lib.Timeout("timed out")

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowTimeoutError):
            wc.run_dwell_workflow("QUJD", retries=1)

    def test_connection_error(self, monkeypatch):
        """ConnectionError raises WorkflowHTTPError mentioning the URL."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
        monkeypatch.setattr(wc.time, "sleep", lambda _: None)
        import requests as req_lib

        def post(*a, **kw):
            raise req_lib.ConnectionError("refused")

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowHTTPError, match="cannot reach"):
            wc.run_dwell_workflow("QUJD", retries=0)

    def test_400_not_retried(self, monkeypatch):
        """4xx (non-auth) raises WorkflowHTTPError immediately without retry."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
        calls = [0]

        def post(*a, **kw):
            calls[0] += 1
            return MockResp(status_code=422)

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowHTTPError):
            wc.run_dwell_workflow("QUJD", retries=2)
        assert calls[0] == 1, "4xx should NOT be retried"

    def test_empty_outputs(self, monkeypatch):
        """Empty outputs list raises WorkflowHTTPError."""
        monkeypatch.setenv("ROBOFLOW_API_KEY", "test")

        def post(*a, **kw):
            return MockResp(data={"outputs": []})

        monkeypatch.setattr(wc.requests, "post", post)
        with pytest.raises(wc.WorkflowHTTPError, match="no 'outputs'"):
            wc.run_dwell_workflow("QUJD")


class TestSummarize:
    def test_full_entry(self):
        entry = FAKE["outputs"][0]
        s = wc.summarize(entry)
        assert s["annotated_image"] == "QUJD"
        assert s["violations"] == 1
        assert s["alert_status"] == "SECURITY ALERT"
        assert s["total_vehicles"] == 1
        assert len(s["vehicles"]) == 1
        assert s["vehicles"][0]["class"] == "car"
        assert s["vehicles"][0]["time_in_zone"] == 6.2

    def test_missing_annotated_image(self):
        entry = dict(FAKE["outputs"][0])
        del entry["annotated_image"]
        s = wc.summarize(entry)
        assert s["annotated_image"] is None

    def test_annotated_image_not_dict(self):
        """If annotated_image is a raw string (not dict), return None gracefully."""
        entry = dict(FAKE["outputs"][0], annotated_image="raw_string")
        s = wc.summarize(entry)
        assert s["annotated_image"] is None

    def test_empty_predictions(self):
        entry = dict(FAKE["outputs"][0], predictions=None)
        s = wc.summarize(entry)
        assert s["vehicles"] == []

    def test_defaults_when_keys_absent(self):
        s = wc.summarize({})
        assert s["violations"] == 0
        assert s["alert_status"] == "OK"
        assert s["total_vehicles"] == 0
        assert s["vehicles"] == []
        assert s["annotated_image"] is None


# ---------------------------------------------------------------------------
# Flask app endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """Return a Flask test client with workflow calls mocked out."""
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test")

    # Patch wc.run_dwell_workflow so tests don't need a live server
    monkeypatch.setattr(wc, "run_dwell_workflow", lambda *a, **kw: FAKE["outputs"][0])

    import app as application
    application.app.config["TESTING"] = True
    with application.app.test_client() as c:
        yield c


class TestFlaskApp:
    def test_index_serves_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"Vehicle Dwell Monitor" in r.data

    def test_frame_happy_path(self, client):
        r = client.post("/api/frame", json={"image": "data:image/jpeg;base64,QUJD"})
        assert r.status_code == 200
        j = r.get_json()
        assert j["violations"] == 1
        assert j["alert_status"] == "SECURITY ALERT"
        assert len(j["vehicles"]) == 1

    def test_frame_missing_image_key(self, client):
        """BUG FIX: missing 'image' key must return 400, not 500."""
        r = client.post("/api/frame", json={"stream_id": "test"})
        assert r.status_code == 400
        assert "image" in r.get_json()["detail"].lower()

    def test_frame_malformed_json(self, client):
        """BUG FIX: malformed JSON body must return 400, not 500."""
        r = client.post("/api/frame", data="not-json",
                        content_type="application/json")
        assert r.status_code == 400

    def test_frame_invalid_numeric_param(self, client):
        """BUG FIX: non-numeric fps must return 400, not 500."""
        r = client.post("/api/frame", json={"image": "QUJD", "fps": "fast"})
        assert r.status_code == 400

    def test_frame_workflow_error_returns_502(self, monkeypatch, client):
        """WorkflowHTTPError from the client must surface as 502."""
        monkeypatch.setattr(wc, "run_dwell_workflow",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                wc.WorkflowHTTPError("server error 503")))
        r = client.post("/api/frame", json={"image": "QUJD"})
        assert r.status_code == 502
        assert r.get_json()["error"] == "WorkflowHTTPError"

    def test_frame_plain_image_b64(self, client):
        """Image without data-URL prefix is handled correctly."""
        r = client.post("/api/frame", json={"image": "QUJD"})
        assert r.status_code == 200

    def test_frame_with_data_url_prefix(self, client):
        """Image with data-URL prefix is stripped correctly."""
        r = client.post("/api/frame",
                        json={"image": "data:image/jpeg;base64,QUJD"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Live integration test (skipped unless env vars set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("ROBOFLOW_API_KEY") and os.environ.get("SAMPLE_IMAGE")),
    reason="set ROBOFLOW_API_KEY and SAMPLE_IMAGE with the local server running",
)
def test_live_local_server():
    b64 = base64.b64encode(Path(os.environ["SAMPLE_IMAGE"]).read_bytes()).decode()
    result = wc.run_dwell_workflow(b64)
    assert wc.EXPECTED_KEYS <= set(result)
