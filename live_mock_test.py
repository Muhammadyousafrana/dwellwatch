"""
live_mock_test.py - End-to-end live mock test for dwell-monitor.

Starts the real Flask app (app.py) in a background thread, patches
workflow_client.run_dwell_workflow with a realistic fake that cycles
through several scenarios, then drives the /api/frame endpoint with
actual HTTP requests and prints a formatted report.

Usage:
    python live_mock_test.py
"""
import base64
import json
import sys
import threading
import time
import unittest.mock as mock
from io import BytesIO
from pathlib import Path

# -- try to import PIL for a real JPEG; fall back to a minimal valid JPEG ------
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import io
import os
import sys

# Force UTF-8 on Windows so ANSI/Unicode chars don't crash cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests as req

# -- colour helpers (ANSI) ----------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}[OK]{RESET}   {msg}")
def fail(msg):print(f"  {RED}[FAIL]{RESET} {msg}"); FAILURES.append(msg)
def info(msg):print(f"  {CYAN}[..]{RESET}   {msg}")
def head(msg):print(f"\n{BOLD}{CYAN}{msg}{RESET}")

FAILURES = []

# -- minimal 1×1 white JPEG as base64 (no PIL needed) ------------------------
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABgUEB"
    "/8QAIRAAAQMEAwEAAAAAAAAAAAAAAQIDBAAFERIhMUH/xAAUAQEAAAAAAAAAAAAAAAAAAAAA"
    "/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8Amst0pC3ZMiC2pt51TYU4EKO"
    "lCiACSBx96UqKUpX/2Q=="
)


def make_jpeg_b64(label="frame", width=320, height=240):
    """Return a base64-encoded JPEG. Uses PIL if available for richer frames."""
    if HAS_PIL:
        img = Image.new("RGB", (width, height), color=(30, 30, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 150, 100], outline=(100, 200, 255), width=2)
        draw.text((15, 50), label, fill=(255, 255, 100))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    return TINY_JPEG_B64


# -- scenario data -------------------------------------------------------------
def _annotated_img(label="annotated"):
    return {"type": "base64", "value": make_jpeg_b64(label)}


SCENARIOS = [
    # 0 - No vehicles, quiet
    {
        "annotated_image": _annotated_img("no vehicles"),
        "violations": 0, "alert_status": "OK", "total_vehicles": 0,
        "predictions": {"predictions": []},
        "slack_notification_status": None, "slack_notification_message": None,
    },
    # 1 - One vehicle, within threshold
    {
        "annotated_image": _annotated_img("1 car"),
        "violations": 0, "alert_status": "OK", "total_vehicles": 1,
        "predictions": {"predictions": [
            {"class": "car", "tracker_id": 1, "time_in_zone": 3.1}]},
        "slack_notification_status": None, "slack_notification_message": None,
    },
    # 2 - Dwell violation triggered
    {
        "annotated_image": _annotated_img("VIOLATION"),
        "violations": 1, "alert_status": "SECURITY ALERT", "total_vehicles": 2,
        "predictions": {"predictions": [
            {"class": "car",   "tracker_id": 1, "time_in_zone": 7.4},
            {"class": "truck", "tracker_id": 2, "time_in_zone": 2.0}]},
        "slack_notification_status": True, "slack_notification_message": "Alert sent to #security",
    },
    # 3 - Alert cooldown; still flagged but Slack not re-sent
    {
        "annotated_image": _annotated_img("still alert"),
        "violations": 1, "alert_status": "SECURITY ALERT", "total_vehicles": 1,
        "predictions": {"predictions": [
            {"class": "car", "tracker_id": 1, "time_in_zone": 9.1}]},
        "slack_notification_status": False, "slack_notification_message": "cooldown",
    },
    # 4 - Clears after alert
    {
        "annotated_image": _annotated_img("clear"),
        "violations": 0, "alert_status": "OK", "total_vehicles": 0,
        "predictions": {"predictions": []},
        "slack_notification_status": None, "slack_notification_message": None,
    },
]

_scenario_idx = 0
_scenario_lock = threading.Lock()

def _fake_workflow(image_b64, **kwargs):
    global _scenario_idx
    with _scenario_lock:
        s = SCENARIOS[_scenario_idx % len(SCENARIOS)]
        _scenario_idx += 1
    time.sleep(0.03)          # simulate small network latency
    return dict(s)            # return a copy


# -- start Flask in a background thread ----------------------------------------
def start_flask():
    import workflow_client as wc
    import app as flask_app

    flask_app.app.config["TESTING"] = False

    # Patch the workflow function globally so every request uses the fake
    with mock.patch.object(wc, "run_dwell_workflow", side_effect=_fake_workflow):
        flask_app.app.run(port=5099, use_reloader=False, threaded=True)


# -- assertion helpers ---------------------------------------------------------
def assert_eq(name, actual, expected):
    if actual == expected:
        ok(f"{name}: {actual!r}")
    else:
        fail(f"{name}: expected {expected!r}, got {actual!r}")

def assert_in(name, key, container):
    if key in container:
        ok(f"{name}: '{key}' present")
    else:
        fail(f"{name}: '{key}' missing from {list(container)[:6]}")

def assert_status(name, resp, expected=200):
    if resp.status_code == expected:
        ok(f"{name}: HTTP {resp.status_code}")
    else:
        fail(f"{name}: expected HTTP {expected}, got {resp.status_code} - {resp.text[:120]}")


# -- test suites ---------------------------------------------------------------
BASE = "http://127.0.0.1:5099"

def test_index():
    head("TEST 1 - GET /  (index page)")
    r = req.get(BASE + "/", timeout=5)
    assert_status("index status", r, 200)
    assert_in("html content-type", "html", r.headers.get("Content-Type",""))
    assert_in("title tag",         "Vehicle Dwell Monitor", r.text)
    assert_in("esc() helper",      "function esc(", r.text)


def test_frame_no_vehicles():
    head("TEST 2 - POST /api/frame  (scenario: no vehicles)")
    payload = {
        "image": "data:image/jpeg;base64," + make_jpeg_b64("test"),
        "stream_id": "mock-stream", "frame_number": 0,
        "fps": 3, "threshold": 5, "cooldown": 5,
    }
    r = req.post(BASE + "/api/frame", json=payload, timeout=10)
    assert_status("frame status", r, 200)
    j = r.json()
    assert_eq("violations",     j["violations"],     0)
    assert_eq("alert_status",   j["alert_status"],   "OK")
    assert_eq("total_vehicles", j["total_vehicles"],  0)
    assert_eq("vehicles count", len(j["vehicles"]),   0)
    assert_in("annotated_image present", "annotated_image", j)


def test_frame_vehicle_ok():
    head("TEST 3 - POST /api/frame  (scenario: 1 car, within threshold)")
    payload = {
        "image": make_jpeg_b64("car"),
        "stream_id": "mock-stream", "frame_number": 1,
        "fps": 3, "threshold": 5, "cooldown": 5,
    }
    r = req.post(BASE + "/api/frame", json=payload, timeout=10)
    assert_status("frame status", r, 200)
    j = r.json()
    assert_eq("violations",            j["violations"],              0)
    assert_eq("total_vehicles",        j["total_vehicles"],          1)
    assert_eq("vehicle class",         j["vehicles"][0]["class"],    "car")
    assert_eq("vehicle tracker_id",    j["vehicles"][0]["tracker_id"], 1)
    assert_eq("vehicle time_in_zone",  j["vehicles"][0]["time_in_zone"], 3.1)


def test_frame_violation():
    head("TEST 4 - POST /api/frame  (scenario: dwell violation)")
    payload = {
        "image": make_jpeg_b64("violation"),
        "stream_id": "mock-stream", "frame_number": 2,
        "fps": 3, "threshold": 5, "cooldown": 5,
    }
    r = req.post(BASE + "/api/frame", json=payload, timeout=10)
    assert_status("frame status", r, 200)
    j = r.json()
    assert_eq("violations",          j["violations"],    1)
    assert_eq("alert_status",        j["alert_status"],  "SECURITY ALERT")
    assert_eq("total_vehicles",      j["total_vehicles"], 2)
    assert_eq("slack sent",          j["slack_notification_status"], True)
    assert_eq("vehicles count",      len(j["vehicles"]), 2)
    assert_eq("car time_in_zone",    j["vehicles"][0]["time_in_zone"], 7.4)
    assert_eq("truck class",         j["vehicles"][1]["class"], "truck")
    # Alert image should be saved
    saved = list(Path("outputs").glob("alert_*.jpg")) if Path("outputs").exists() else []
    if saved:
        ok(f"alert image saved: {saved[-1].name}")
    else:
        info("alert image not yet saved (cooldown may still be active)")


def test_frame_cooldown():
    head("TEST 5 - POST /api/frame  (scenario: alert cooldown)")
    payload = {
        "image": make_jpeg_b64("cooldown"),
        "stream_id": "mock-stream", "frame_number": 3,
        "fps": 3, "threshold": 5, "cooldown": 5,
    }
    r = req.post(BASE + "/api/frame", json=payload, timeout=10)
    assert_status("frame status", r, 200)
    j = r.json()
    assert_eq("violations",          j["violations"],    1)
    assert_eq("alert_status",        j["alert_status"],  "SECURITY ALERT")
    assert_eq("slack NOT re-sent",   j["slack_notification_status"], False)


def test_frame_clears():
    head("TEST 6 - POST /api/frame  (scenario: clears after alert)")
    payload = {
        "image": make_jpeg_b64("clear"),
        "stream_id": "mock-stream", "frame_number": 4,
        "fps": 3, "threshold": 5, "cooldown": 5,
    }
    r = req.post(BASE + "/api/frame", json=payload, timeout=10)
    assert_status("frame status", r, 200)
    j = r.json()
    assert_eq("violations cleared", j["violations"],    0)
    assert_eq("status back to OK",  j["alert_status"],  "OK")


def test_bad_requests():
    head("TEST 7 - Error handling (bad requests → 400)")

    # 7a: missing body
    r = req.post(BASE + "/api/frame", data="", content_type="application/json", timeout=5)
    assert_status("missing body → 400", r, 400)

    # 7b: missing 'image' key
    r = req.post(BASE + "/api/frame", json={"stream_id": "x"}, timeout=5)
    assert_status("missing image key → 400", r, 400)
    j = r.json()
    assert_in("error field present", "error", j)
    assert_in("detail field present", "detail", j)

    # 7c: bad numeric param
    r = req.post(BASE + "/api/frame",
                 json={"image": "QUJD", "fps": "very_fast"}, timeout=5)
    assert_status("bad fps → 400", r, 400)

    # 7d: completely invalid JSON
    r = req.post(BASE + "/api/frame", data="{{not json}}",
                 headers={"Content-Type": "application/json"}, timeout=5)
    assert_status("malformed JSON → 400", r, 400)


def test_concurrent_frames():
    head("TEST 8 - Concurrency (5 simultaneous frame requests)")
    import concurrent.futures

    def send(n):
        return req.post(BASE + "/api/frame", timeout=10, json={
            "image": make_jpeg_b64(f"concurrent-{n}"),
            "stream_id": "concurrent-stream",
            "frame_number": n, "fps": 3, "threshold": 5, "cooldown": 5,
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(send, i) for i in range(5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    codes = [r.status_code for r in results]
    if all(c == 200 for c in codes):
        ok(f"all 5 concurrent requests returned 200")
    else:
        fail(f"concurrent requests returned: {codes}")

    # All must have valid JSON with expected shape
    for r in results:
        j = r.json()
        for key in ("violations", "alert_status", "total_vehicles", "vehicles", "annotated_image"):
            if key not in j:
                fail(f"concurrent response missing key: {key}")
                break
        else:
            ok(f"response shape valid (violations={j['violations']}, status={j['alert_status']})")


def test_stream_continuity():
    head("TEST 9 - Stream continuity (10 sequential frames, incrementing frame_number)")
    frame_numbers_sent = []
    for i in range(10):
        frame_numbers_sent.append(i)
        r = req.post(BASE + "/api/frame", timeout=10, json={
            "image": make_jpeg_b64(f"seq-{i}"),
            "stream_id": "continuity-stream",
            "frame_number": i, "fps": 3, "threshold": 5, "cooldown": 5,
        })
        if r.status_code != 200:
            fail(f"frame {i} returned {r.status_code}")
            break
    else:
        ok(f"all {len(frame_numbers_sent)} sequential frames processed without error")


# -- main ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  DWELL MONITOR - LIVE MOCK END-TO-END TEST{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  PIL available: {HAS_PIL} {'(rich JPEG frames)' if HAS_PIL else '(using minimal JPEG)'}")

    # Start Flask in background thread
    t = threading.Thread(target=start_flask, daemon=True)
    t.start()

    # Wait for server to be ready
    info("Starting Flask on :5099 ...")
    for _ in range(30):
        try:
            req.get(BASE + "/", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        print(f"{RED}Server did not start in time!{RESET}")
        sys.exit(1)
    ok("Server is up")

    # Run all tests
    test_index()
    test_frame_no_vehicles()
    test_frame_vehicle_ok()
    test_frame_violation()
    test_frame_cooldown()
    test_frame_clears()
    test_bad_requests()
    test_concurrent_frames()
    test_stream_continuity()

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    if not FAILURES:
        print(f"{BOLD}{GREEN}  ALL CHECKS PASSED [OK]{RESET}")
    else:
        print(f"{BOLD}{RED}  {len(FAILURES)} FAILURE(S):{RESET}")
        for f in FAILURES:
            print(f"    {RED}[FAIL]{RESET} {f}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    sys.exit(0 if not FAILURES else 1)
