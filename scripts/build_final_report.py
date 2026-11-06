"""
build_final_report.py
=====================
Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.md dosyasini okur ve
ayni ada sahip .docx + .pdf dosyalarini uretir.

Word (docx) uretiminde python-docx, PDF uretiminde reportlab kullanilir.

Kullanim:
    python scripts/build_final_report.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Turkce karakterleri destekleyen Unicode fontlari yukle (Windows Calibri ailesi)
_FONT_DIR = Path(r"C:\Windows\Fonts")
_FONTS = {
    "Report": _FONT_DIR / "calibri.ttf",
    "Report-Bold": _FONT_DIR / "calibrib.ttf",
    "Report-Italic": _FONT_DIR / "calibrii.ttf",
    "Report-Mono": _FONT_DIR / "consola.ttf",  # Consolas
}
for name, path in _FONTS.items():
    if path.exists():
        try:
            pdfmetrics.registerFont(TTFont(name, str(path)))
        except Exception as e:
            print(f"[WARN] font register hatasi {name}: {e}")
    else:
        print(f"[WARN] font bulunamadi: {path}")

# Bold/italic eslestirmesi
pdfmetrics.registerFontFamily(
    "Report",
    normal="Report",
    bold="Report-Bold",
    italic="Report-Italic",
    boldItalic="Report-Bold",
)


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.md"
DOCX_PATH = ROOT / "Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.docx"
PDF_PATH = ROOT / "Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.pdf"


# ------------------------------------------------------------
# Markdown parser: cok yalin, bu dokuman icin yeterli.
# Kabul edilen bloklar:
#   - # / ## / ### heading
#   - Paragraf
#   - | tablolari
#   - - bullet list
#   - --- (horizontal rule / bolum ayraci)
# ------------------------------------------------------------


def parse_md(text: str):
    lines = text.splitlines()
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}\s*$", line):
            blocks.append(("hr", None))
            i += 1
            continue

        # Table (| ... |)
        if line.lstrip().startswith("|"):
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            # parse
            parsed = []
            for r in rows:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                # seperator row mu? (--- iceriyor)
                if all(re.match(r"^:?-+:?$", c or "-") for c in cells):
                    continue
                parsed.append(cells)
            if parsed:
                blocks.append(("table", parsed))
            continue

        # Bullet list
        if line.lstrip().startswith("- "):
            items = []
            while i < n and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            blocks.append(("ul", items))
            continue

        # Code block (```)
        if line.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            blocks.append(("code", "\n".join(buf)))
            continue

        # Paragraph: collect until blank line
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() and not lines[i].startswith(
            ("#", "|", "- ", "```", "---")
        ):
            para_lines.append(lines[i].rstrip())
            i += 1
        blocks.append(("p", " ".join(para_lines)))

    return blocks


# ------------------------------------------------------------
# Inline markdown: **bold**, *italic*, `code`
# Word ve PDF icin farkli render yontemleri var; burada basit
# bir token stream uretiyoruz.
# ------------------------------------------------------------

INLINE_RE = re.compile(
    r"(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`)"
)


def inline_tokens(text: str):
    """Metni (style, text) parcalarina boler.
    style in {'bold','italic','code','plain'}.
    """
    out = []
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(("plain", text[pos : m.start()]))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append(("bold", tok[2:-2]))
        elif tok.startswith("*"):
            out.append(("italic", tok[1:-1]))
        elif tok.startswith("`"):
            out.append(("code", tok[1:-1]))
        pos = m.end()
    if pos < len(text):
        out.append(("plain", text[pos:]))
    return out


# ============================================================
# DOCX generator
# ============================================================


def _add_runs(paragraph, tokens):
    for style, t in tokens:
        run = paragraph.add_run(t)
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        if style == "bold":
            run.bold = True
        elif style == "italic":
            run.italic = True
        elif style == "code":
            run.font.name = "Consolas"
            run.font.size = Pt(10)


def build_docx(blocks, out_path: Path):
    doc = Document()

    # Varsayilan stil boyut
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    # Sayfa kenar boslugu
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    for kind, content in blocks:
        if kind == "h1":
            p = doc.add_heading(content, level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif kind == "h2":
            doc.add_heading(content, level=2)
        elif kind == "h3":
            doc.add_heading(content, level=3)
        elif kind == "p":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(8)
            _add_runs(p, inline_tokens(content))
        elif kind == "ul":
            for item in content:
                p = doc.add_paragraph(style="List Bullet")
                _add_runs(p, inline_tokens(item))
        elif kind == "table":
            header = content[0]
            body = content[1:]
            tbl = doc.add_table(rows=1 + len(body), cols=len(header))
            tbl.style = "Light Grid Accent 1"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            # header
            hdr_cells = tbl.rows[0].cells
            for j, cell_text in enumerate(header):
                hdr_cells[j].text = ""
                p = hdr_cells[j].paragraphs[0]
                run = p.add_run(cell_text)
                run.bold = True
                run.font.size = Pt(10)
            # body
            for i_row, row in enumerate(body, 1):
                cells = tbl.rows[i_row].cells
                for j, cell_text in enumerate(row):
                    if j >= len(cells):
                        break
                    cells[j].text = ""
                    para = cells[j].paragraphs[0]
                    _add_runs(para, inline_tokens(cell_text))
                    for run in para.runs:
                        run.font.size = Pt(10)
            doc.add_paragraph()  # sonrasinda bosluk
        elif kind == "code":
            p = doc.add_paragraph()
            run = p.add_run(content)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif kind == "hr":
            # Yatay ayraci sayfa kirpma ile yakinlatir
            doc.add_paragraph("_" * 60)

    doc.save(str(out_path))


# ============================================================
# PDF generator (ReportLab)
# ============================================================


def _rl_escape(text: str) -> str:
    # reportlab Paragraph icin temel XML kacisi
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_MONO_FONT_NAME = (
    "Report-Mono" if Path(r"C:\Windows\Fonts\consola.ttf").exists() else "Courier"
)


def _inline_to_rl(text: str) -> str:
    """Inline markdown -> reportlab paragraph HTML."""
    toks = inline_tokens(text)
    out = []
    for style, t in toks:
        esc = _rl_escape(t)
        if style == "bold":
            out.append(f"<b>{esc}</b>")
        elif style == "italic":
            out.append(f"<i>{esc}</i>")
        elif style == "code":
            out.append(f'<font face="{_MONO_FONT_NAME}">{esc}</font>')
        else:
            out.append(esc)
    return "".join(out)


def build_pdf(blocks, out_path: Path):
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        title="Bitirme 2 — Final Raporu",
        author="Ahmet Dulger, Muhammed Esat Demir",
    )

    ss = getSampleStyleSheet()

    base_font = "Report"
    bold_font = "Report-Bold"
    mono_font = "Report-Mono" if Path(r"C:\Windows\Fonts\consola.ttf").exists() else "Courier"

    styles = {
        "h1": ParagraphStyle(
            "h1",
            parent=ss["Heading1"],
            fontName=bold_font,
            fontSize=18,
            textColor=colors.HexColor("#1F3864"),
            spaceAfter=12,
            spaceBefore=18,
            leading=22,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=ss["Heading2"],
            fontName=bold_font,
            fontSize=14,
            textColor=colors.HexColor("#1F3864"),
            spaceAfter=8,
            spaceBefore=14,
            leading=18,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=ss["Heading3"],
            fontName=bold_font,
            fontSize=11.5,
            textColor=colors.HexColor("#2E5FA0"),
            spaceAfter=6,
            spaceBefore=10,
            leading=15,
            keepWithNext=True,
        ),
        "p": ParagraphStyle(
            "p",
            parent=ss["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
            alignment=4,  # justify
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=ss["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=13,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=3,
            alignment=4,
        ),
        "code": ParagraphStyle(
            "code",
            parent=ss["BodyText"],
            fontName=mono_font,
            fontSize=8.5,
            leading=11,
            leftIndent=10,
            spaceAfter=8,
            backColor=colors.HexColor("#F4F4F4"),
        ),
    }

    flowables = []

    for kind, content in blocks:
        if kind in ("h1", "h2", "h3"):
            flowables.append(Paragraph(_inline_to_rl(content), styles[kind]))
        elif kind == "p":
            flowables.append(Paragraph(_inline_to_rl(content), styles["p"]))
        elif kind == "ul":
            for item in content:
                flowables.append(
                    Paragraph("• " + _inline_to_rl(item), styles["bullet"])
                )
        elif kind == "code":
            for line in content.split("\n"):
                flowables.append(Paragraph(_rl_escape(line) or "&nbsp;", styles["code"]))
            flowables.append(Spacer(1, 6))
        elif kind == "hr":
            flowables.append(Spacer(1, 6))
            flowables.append(
                Table(
                    [[""]],
                    colWidths=[doc.width],
                    rowHeights=[1],
                    style=TableStyle(
                        [("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.grey)]
                    ),
                )
            )
            flowables.append(Spacer(1, 6))
        elif kind == "table":
            # ilk satir header
            rows_rl = []
            for r_i, row in enumerate(content):
                rl_row = [
                    Paragraph(_inline_to_rl(cell), styles["p"])
                    for cell in row
                ]
                rows_rl.append(rl_row)

            # sutun genisligi: esit dagit
            ncols = max(len(r) for r in content)
            col_w = doc.width / ncols

            t = Table(rows_rl, colWidths=[col_w] * ncols, repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), bold_font),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F4F7FB")],
                        ),
                    ]
                )
            )
            flowables.append(t)
            flowables.append(Spacer(1, 10))

    doc.build(flowables)


# ============================================================
# Main
# ============================================================


def main():
    text = MD_PATH.read_text(encoding="utf-8")
    blocks = parse_md(text)
    print(f"[INFO] parsed {len(blocks)} blocks from {MD_PATH.name}")

    print(f"[INFO] writing {DOCX_PATH.name} ...")
    build_docx(blocks, DOCX_PATH)
    print(f"[OK] {DOCX_PATH}")

    print(f"[INFO] writing {PDF_PATH.name} ...")
    build_pdf(blocks, PDF_PATH)
    print(f"[OK] {PDF_PATH}")


if __name__ == "__main__":
    main()
