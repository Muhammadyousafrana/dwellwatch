import base64
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import workflow_client as wc

app = Flask(__name__, static_folder="static")
OUT_DIR = Path("outputs")
_last_save = 0.0
_save_lock = threading.Lock()  # BUG FIX: prevent concurrent writes from racing


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/frame")
def frame():
    global _last_save

    # BUG FIX: validate that JSON body parsed correctly before accessing keys
    d = request.get_json(force=True, silent=True)
    if not d or "image" not in d:
        return jsonify(error="BadRequest", detail="JSON body with 'image' field required"), 400

    b64 = d["image"].split(",")[-1]

    # BUG FIX: wrap type coercions to return 400 instead of crashing with 500
    try:
        frame_number = int(d.get("frame_number", 0))
        fps = float(d.get("fps", 3))
        dwell_threshold = float(d.get("threshold", 5))
        cooldown = float(d.get("cooldown", 5))
    except (ValueError, TypeError) as e:
        return jsonify(error="BadRequest", detail=f"Invalid numeric parameter: {e}"), 400

    try:
        entry = wc.run_dwell_workflow(
            b64,
            stream_id=d.get("stream_id", "stream-1"),
            frame_number=frame_number,
            fps=fps,
            dwell_threshold_seconds=dwell_threshold,
            alert_cooldown_seconds=cooldown,
        )
    except wc.WorkflowError as e:
        return jsonify(error=type(e).__name__, detail=str(e)), 502

    out = wc.summarize(entry)

    # BUG FIX: use a lock to prevent two concurrent requests from both passing
    # the time check and writing simultaneously (race condition)
    if out["violations"] and out["annotated_image"]:
        with _save_lock:
            now = time.time()
            if now - _last_save > 5:
                OUT_DIR.mkdir(exist_ok=True)
                (OUT_DIR / f"alert_{int(now)}.jpg").write_bytes(
                    base64.b64decode(out["annotated_image"])
                )
                _last_save = now

    return jsonify(out)


if __name__ == "__main__":
    # debug=False by default when not in development; set DEBUG=1 env var to enable
    import os
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(port=5000, debug=debug)
