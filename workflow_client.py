"""Client for the 'Vehicle Dwell Time Monitor v2' Roboflow Workflow (local inference server)."""
import os
import time
from datetime import datetime, timezone

import requests

WORKSPACE = "alis-workspace-zktpw"
WORKFLOW_ID = "vehicle-dwell-time-monitor-v2-1777526043769"
EXPECTED_KEYS = {"annotated_image", "violations", "alert_status", "total_vehicles",
                 "predictions", "slack_notification_status", "slack_notification_message"}


class WorkflowError(Exception):
    """Base error for workflow calls."""


class WorkflowAuthError(WorkflowError):
    """401/403 from the inference server."""


class WorkflowTimeoutError(WorkflowError):
    """Request timed out after all retries."""


class WorkflowHTTPError(WorkflowError):
    """Non-retryable or exhausted HTTP/connection failure."""


def _api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise WorkflowAuthError("ROBOFLOW_API_KEY is not set")
    return key


def run_dwell_workflow(image_b64, *, stream_id="stream-1", frame_number=0, fps=3.0,
                       dwell_threshold_seconds=5, alert_cooldown_seconds=5,
                       active_learning_project="vehicle-dwell-monitor",
                       api_url=None, timeout=30, retries=2, session=None):
    """Run one frame through the workflow and return the output dict for that frame.

    video_metadata (same stream_id on every frame) lets the server's tracker and
    time_in_zone blocks keep state across frames; without it dwell time stays ~0.

    The video_metadata is passed as a top-level workflow input (not nested inside
    the image object) so the Roboflow inference server can route it correctly.
    """
    base_url = (api_url or os.environ.get("INFERENCE_API_URL", "http://localhost:9001")).rstrip("/")
    url = f"{base_url}/{WORKSPACE}/workflows/{WORKFLOW_ID}"

    # BUG FIX: video_metadata must be a top-level input, not nested inside the image dict.
    # BUG FIX: Roboflow inference server authenticates via ?api_key= query param, not Bearer token.
    body = {
        "api_key": _api_key(),
        "inputs": {
            "image": {"type": "base64", "value": image_b64},
            "video_metadata": {
                "video_identifier": stream_id,
                "frame_number": frame_number,
                "fps": fps,
                "frame_timestamp": datetime.now(timezone.utc).isoformat(),
                "comes_from_video_file": False,
            },
            "dwell_threshold_seconds": dwell_threshold_seconds,
            "alert_cooldown_seconds": alert_cooldown_seconds,
            "active_learning_project": active_learning_project,
        },
    }
    headers = {"Content-Type": "application/json"}
    post = (session or requests).post
    last = None
    for attempt in range(retries + 1):
        try:
            r = post(url, json=body, headers=headers, timeout=timeout)
            if r.status_code in (401, 403):
                raise WorkflowAuthError(f"auth rejected ({r.status_code}): {r.text[:200]}")
            if r.status_code >= 500:
                last = WorkflowHTTPError(f"server error {r.status_code}: {r.text[:200]}")
            elif r.status_code >= 400:
                raise WorkflowHTTPError(f"{r.status_code}: {r.text[:200]}")
            else:
                outputs = r.json().get("outputs") or []
                if not outputs:
                    raise WorkflowHTTPError("response has no 'outputs'")
                return outputs[0]
        except requests.Timeout as e:
            last = WorkflowTimeoutError(str(e))
        except requests.ConnectionError as e:
            last = WorkflowHTTPError(f"cannot reach {url}: {e}")
        if attempt < retries:
            time.sleep(0.5 * 2 ** attempt)

    # BUG FIX: Guard against last being None (should not happen, but prevents a
    # confusing "TypeError: exceptions must derive from BaseException" crash).
    if last is None:
        last = WorkflowHTTPError("all retries exhausted with no recorded error")
    raise last


def summarize(entry):
    """Reduce a raw output entry to the small set of fields the app uses."""
    img = entry.get("annotated_image")
    preds = (entry.get("predictions") or {}).get("predictions") or []
    return {
        "annotated_image": img.get("value") if isinstance(img, dict) else None,
        "violations": entry.get("violations", 0),
        "alert_status": entry.get("alert_status", "OK"),
        "total_vehicles": entry.get("total_vehicles", 0),
        "slack_notification_status": entry.get("slack_notification_status"),
        "slack_notification_message": entry.get("slack_notification_message"),
        "vehicles": [
            {
                "class": p.get("class"),
                "tracker_id": p.get("tracker_id"),
                "time_in_zone": p.get("time_in_zone"),
            }
            for p in preds
        ],
    }
