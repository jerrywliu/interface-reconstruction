"""
Slack integration test.

This test verifies that results can be sent to Slack.
Run manually with:
  SLACK_BOT_TOKEN=... SLACK_CHANNEL=... python -m test.integration.test_slack_integration
or:
  SLACK_WEBHOOK_URL=... python -m test.integration.test_slack_integration

This test also reads optional local config from:
  config/local/slack.env
"""

import os
import json
import tempfile
from datetime import datetime

from util.io.slack import send_results_to_slack


def _build_message():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Slack integration test message ({stamp})"


def test_slack_integration():
    """
    Test sending results to Slack with a message, an image, and a data dump.
    """
    tmp_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write("Slack integration test text file.\n")
            tmp_path = tmp.name
            tmp_paths.append(tmp_path)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            json.dump({"hello": "slack", "ok": True}, tmp)
            tmp_path = tmp.name
            tmp_paths.append(tmp_path)

        try:
            from PIL import Image
        except Exception:
            Image = None

        if Image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img_path = tmp.name
            img = Image.new("RGB", (240, 120), color=(220, 235, 255))
            img.save(img_path)
            tmp_paths.append(img_path)

        ok = send_results_to_slack(_build_message(), tmp_paths)
        if not ok:
            raise RuntimeError("Slack send failed (see console output for details).")
    finally:
        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    test_slack_integration()
    print("Slack integration test completed.")
