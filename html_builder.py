"""Builds the self-contained interactive HTML dashboard."""
import json
import os

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "html_template.html")


def build_html(data, out_path):
    with open(_TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA_JSON__*/null", payload)
    assert "__DATA_JSON__" not in html, "data placeholder not replaced"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] {out_path} written ({len(html) / 1024:.0f} KB)")
