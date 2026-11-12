"""Minimal ComfyUI API client used by the batch runners.

Tested against the documented /prompt, /history, /view and /upload/image
endpoints. The API workflow must be exported from ComfyUI with "Export (API)".

Node titles are the stable contract between the UI workflow and this client:
  POSITIVE / NEGATIVE / REF1 / REF2 / START_IMAGE / AUDIO / SAVE
"""
from __future__ import annotations

import json
import mimetypes
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import requests


class ComfyError(RuntimeError):
    pass


SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
SEED_NODES = SAMPLER_TYPES | {"RandomNoise"}


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", input_dir: str | Path | None = None):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.input_dir = Path(input_dir) if input_dir else None

    # ------------------------------------------------------------------ http
    def _get(self, path: str, **kwargs) -> requests.Response:
        r = requests.get(self.base_url + path, timeout=kwargs.pop("timeout", 60), **kwargs)
        r.raise_for_status()
        return r

    def _post(self, path: str, **kwargs) -> requests.Response:
        r = requests.post(self.base_url + path, timeout=kwargs.pop("timeout", 120), **kwargs)
        r.raise_for_status()
        return r

    def system_stats(self) -> dict:
        return self._get("/system_stats").json()

    def queue_prompt(self, workflow: dict) -> str:
        try:
            r = self._post("/prompt", json={"prompt": workflow, "client_id": self.client_id})
        except requests.HTTPError as e:
            body = e.response.text if e.response is not None else ""
            code = e.response.status_code if e.response is not None else "?"
            raise ComfyError(f"queue rejected (HTTP {code}): {body[:4000]}") from e
        data = r.json()
        if "error" in data or data.get("node_errors"):
            raise ComfyError("queue rejected: " + json.dumps(data, ensure_ascii=False)[:4000])
        pid = data.get("prompt_id")
        if not pid:
            raise ComfyError("no prompt_id in response: " + json.dumps(data)[:1000])
        return pid

    def history(self, prompt_id: str) -> dict:
        data = self._get(f"/history/{prompt_id}").json()
        return data.get(prompt_id, {})

    def wait(self, prompt_id: str, timeout: float = 7200, poll: float = 2.0) -> dict:
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            entry = self.history(prompt_id)
            if entry:
                status = entry.get("status", {}) or {}
                if status.get("status_str") == "error":
                    raise ComfyError("execution error: " + json.dumps(status, ensure_ascii=False)[:4000])
                if status.get("completed") or entry.get("outputs"):
                    return entry
                last = status
            time.sleep(poll)
        raise ComfyError(f"timeout after {timeout}s; last status={last}")

    def interrupt(self) -> None:
        try:
            self._post("/interrupt")
        except Exception:
            pass

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        try:
            self._post("/free", json={"unload_models": unload_models, "free_memory": free_memory})
        except Exception:
            pass

    # ----------------------------------------------------------------- files
    def stage_file(self, src: str | Path, prefix: str = "") -> str:
        """Copy a local file into ComfyUI/input and return the name used by Load* nodes.

        Copying is more reliable than /upload/image for audio and for paths with
        non-ASCII characters. If input_dir is not configured, fall back to HTTP.
        """
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        if self.input_dir:
            self.input_dir.mkdir(parents=True, exist_ok=True)
            name = f"{prefix}{src.stem}{src.suffix}" if prefix else src.name
            dst = self.input_dir / name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            return name
        # remote fallback
        mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        with open(src, "rb") as f:
            r = self._post("/upload/image", files={"image": (src.name, f, mime)},
                           data={"overwrite": "true", "type": "input"})
        data = r.json()
        return data.get("name", src.name)

    def download_outputs(self, entry: dict, out_dir: str | Path, prefix: str | None = None) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        outputs = entry.get("outputs", {}) or {}
        for _node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for key in ("images", "gifs", "videos", "audio", "files"):
                items = node_out.get(key) or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    params = {
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                    r = self._get("/view", params=params, timeout=600)
                    ext = Path(item["filename"]).suffix or ".bin"
                    if prefix:
                        name = f"{prefix}{ext}" if len(saved) == 0 else f"{prefix}_{len(saved)+1}{ext}"
                    else:
                        name = item["filename"]
                    dst = out_dir / name
                    dst.write_bytes(r.content)
                    saved.append(dst)
        return saved

    def run(self, workflow: dict, out_dir: str | Path, prefix: str | None = None,
            timeout: float = 7200) -> list[Path]:
        pid = self.queue_prompt(workflow)
        entry = self.wait(pid, timeout=timeout)
        return self.download_outputs(entry, out_dir, prefix=prefix)

    # --------------------------------------------------------------- workflow
    @staticmethod
    def nodes(workflow: dict) -> Iterable[tuple[str, dict]]:
        for node_id, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                yield node_id, node

    @classmethod
    def titles(cls, workflow: dict) -> list[tuple[str, str, str]]:
        out = []
        for node_id, node in cls.nodes(workflow):
            meta = node.get("_meta") or {}
            out.append((meta.get("title", ""), node.get("class_type", ""), node_id))
        return out

    @classmethod
    def patch(cls, workflow: dict, title: str, input_name: str, value: Any,
              required: bool = True) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            meta = node.get("_meta") or {}
            if meta.get("title") == title:
                node.setdefault("inputs", {})[input_name] = value
                hit += 1
        if hit == 0 and required:
            available = ", ".join(f"{t}({c})" for t, c, _ in cls.titles(workflow))
            raise ComfyError(f"node title {title!r} not found. Available: {available}")
        return hit

    @classmethod
    def patch_any(cls, workflow: dict, title: str, input_names: list[str], value: Any,
                  required: bool = True) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            meta = node.get("_meta") or {}
            if meta.get("title") != title:
                continue
            inputs = node.setdefault("inputs", {})
            for name in input_names:
                if name in inputs:
                    inputs[name] = value
                    hit += 1
                    break
            else:
                inputs[input_names[0]] = value
                hit += 1
        if hit == 0 and required:
            available = ", ".join(f"{t}({c})" for t, c, _ in cls.titles(workflow))
            raise ComfyError(f"node title {title!r} not found. Available: {available}")
        return hit

    @classmethod
    def set_seed(cls, workflow: dict, seed: int) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            if node.get("class_type") not in SEED_NODES:
                continue
            inputs = node.setdefault("inputs", {})
            if "seed" in inputs:
                inputs["seed"] = seed
                hit += 1
            elif "noise_seed" in inputs:
                inputs["noise_seed"] = seed
                hit += 1
        return hit

    @classmethod
    def set_steps_cfg(cls, workflow: dict, steps: int | None, cfg: float | None) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            if node.get("class_type") not in SAMPLER_TYPES:
                continue
            inputs = node.setdefault("inputs", {})
            if steps is not None and "steps" in inputs:
                inputs["steps"] = steps
            if cfg is not None and "cfg" in inputs:
                inputs["cfg"] = cfg
            hit += 1
        return hit

    @classmethod
    def set_size(cls, workflow: dict, width: int | None, height: int | None) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            ct = node.get("class_type", "")
            inputs = node.setdefault("inputs", {})
            if "width" not in inputs or "height" not in inputs:
                continue
            if "Latent" not in ct and not ct.startswith("Empty"):
                continue
            if width is not None:
                inputs["width"] = width
            if height is not None:
                inputs["height"] = height
            hit += 1
        return hit

    @classmethod
    def set_prefix(cls, workflow: dict, prefix: str) -> int:
        hit = 0
        for _node_id, node in cls.nodes(workflow):
            if node.get("class_type") in ("SaveImage", "VHS_VideoCombine", "SaveVideo"):
                node.setdefault("inputs", {})["filename_prefix"] = prefix
                hit += 1
        return hit

    @classmethod
    def set_prompts(cls, workflow: dict, positive: str, negative: str | None) -> None:
        cls.patch_any(workflow, "POSITIVE", ["text", "prompt"], positive)
        if negative is not None:
            cls.patch_any(workflow, "NEGATIVE", ["text", "prompt"], negative)
