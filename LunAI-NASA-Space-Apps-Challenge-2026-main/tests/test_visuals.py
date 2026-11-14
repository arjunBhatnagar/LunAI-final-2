import json
import re

from lunai.visuals import build_viewer_html


def test_viewer_inlines_assets_and_escapes_config():
    page = build_viewer_html({"simulation": {}, "note": "</script><b>"})
    assert "/*__" not in page
    assert "<\\/script><b>" in page
    assert page.count("</script>") == 3  # two CDN scripts + the inline app script
    config = re.search(r"const cfg = (\{.*?\});\n", page).group(1)
    assert json.loads(config.replace("<\\/", "</"))["note"] == "</script><b>"
