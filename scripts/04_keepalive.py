#!/usr/bin/env python3
"""Tiny heartbeat for long ModelScope notebook sessions.

It pings ComfyUI's /system_stats every N seconds and appends a line to the log.
This keeps a process active, but it may NOT count as user activity on the
platform. Keep the JupyterLab tab open as well, and stop the instance manually
when you leave for more than ~10 minutes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request


def ping(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(200)
            return True, body.decode("utf-8", "replace")[:120]
    except Exception as e:  # noqa: BLE001
        return False, repr(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188/system_stats")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--log", default="logs/keepalive.log")
    args = ap.parse_args()

    print(f"keepalive: {args.url} every {args.interval}s -> {args.log}")
    try:
        while True:
            ok, detail = ping(args.url)
            ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            line = json.dumps({"ts": ts, "ok": ok, "detail": detail}, ensure_ascii=False)
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)
            time.sleep(max(5, args.interval))
    except KeyboardInterrupt:
        print("keepalive stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
