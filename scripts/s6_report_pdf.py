"""S6: render reports/technical_report.md to a two-page A4 HTML and PDF.

Markdown -> HTML with Python-Markdown (tables extension), a print stylesheet sized for two A4
pages, then headless Chrome or Edge --print-to-pdf. Figures are embedded as data URIs so the
HTML is self-contained.

Run: uv run --with markdown python scripts/s6_report_pdf.py
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "technical_report.md"
HTML = ROOT / "reports" / "technical_report.html"
PDF = ROOT / "reports" / "technical_report.pdf"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "google-chrome", "chromium", "chromium-browser"]

CSS = """
@page { size: A4; margin: 10mm 12mm 10mm 12mm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 8.3pt;
       line-height: 1.32; color: #0b0b0b; background: #fff; }
body { margin: 0; }
h1 { font-size: 13.5pt; margin: 0 0 1mm; line-height: 1.2; }
.byline { color: #52514e; margin: 0 0 2.5mm; font-size: 8pt; }
h2 { font-size: 9.8pt; margin: 2.6mm 0 1mm; color: #1f4f8f; }
p { margin: 0 0 1.4mm; }
ul, ol { margin: 0 0 1.4mm; padding-left: 4.5mm; }
li { margin: 0 0 0.3mm; }
table { border-collapse: collapse; width: 100%; margin: 1mm 0 1.6mm; font-size: 7.9pt; }
th, td { border-bottom: 0.4pt solid #cfcdc8; padding: 0.7mm 1.2mm; text-align: left; }
th { background: #f1f0ec; font-weight: 600; }
img { display: block; margin: 1mm auto 1.6mm; max-width: 100%; }
img[alt^="Recall"] { width: 80%; }
img[alt^="Threshold"] { width: 44%; }
strong { font-weight: 650; }
"""


def embed(html: str) -> str:
    def repl(m):
        p = (SRC.parent / m.group(1)).resolve()
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return re.sub(r'src="([^"]+\.png)"', repl, html)


LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def blank_before_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph (GitHub
    does not); add it."""
    out, prev = [], ""
    for line in text.splitlines():
        if LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith(" "):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out) + "\n"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        head, text = text[3:].split("---", 1)
        for line in head.strip().splitlines():
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    body = markdown.markdown(blank_before_lists(text), extensions=["tables"])
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{meta.get('title', 'EchoFind')}</title><style>{CSS}</style></head><body>"
            f"<h1>{meta.get('title', '')}</h1><p class='byline'>{meta.get('author', '')}</p>"
            f"{embed(body)}</body></html>")
    HTML.write_text(html, encoding="utf-8")
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        sys.exit("no Chrome/Edge found; open the HTML and print to PDF")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", HTML.name, PDF.name, f"{PDF.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
