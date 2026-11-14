"""Assemble the self-contained Three.js viewer page from ``lunai/web``."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WEB_DIR = Path(__file__).with_name("web")


def build_viewer_html(config: Mapping[str, Any]) -> str:
    """Inline the stylesheet, script and JSON config into one HTML document."""
    template = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    # "</" must not appear inside an inline <script>, or the browser ends the script early.
    config_json = json.dumps(config, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    script = script.replace("/*__CONFIG__*/null", config_json, 1)
    return template.replace("/*__STYLES__*/", styles, 1).replace("/*__SCRIPT__*/", script, 1)
