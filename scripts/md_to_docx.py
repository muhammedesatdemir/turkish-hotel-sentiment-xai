# -*- coding: utf-8 -*-
"""
Basit ama saglam Markdown -> DOCX donusturucu (bu rapora ozel).
Destekler: # ## ### basliklar, paragraf, **bold**, `code`, tablolar (| |),
fenced kod bloklari (```), gomulu gorseller ![](yol), liste ( - / 1. ),
yatay cizgi (---).
"""
import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "dosyalar" / "Bitirme2_FinalRapor_YENI_AhmetDulger_MuhammedEsatDemir.md"
OUT = ROOT / "dosyalar" / "Bitirme2_FinalRapor_YENI_AhmetDulger_MuhammedEsatDemir.docx"
BASE = MD.parent  # gorsel yollari bu klasore goreli (../reports/..)

doc = Document()
# Varsayilan font
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def add_runs_with_inline(par, text):
    """**bold** ve `code` inline bicimlerini isler."""
    # token: **...** veya `...` veya duz
    parts = re.split(r"(\*\*.+?\*\*|`.+?`)", text)
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            r = par.add_run(p[2:-2]); r.bold = True
        elif p.startswith("`") and p.endswith("`"):
            r = par.add_run(p[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
        else:
            par.add_run(p)


def flush_table(rows):
    """Markdown tablo satirlarini DOCX tablosuna cevirir."""
    # ayirici satiri (---) at
    cleaned = [r for r in rows if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", r) or "|" not in r or not set(r.replace("|", "").strip()) <= set("-: ")]
    # daha basit: ikinci satir genelde ayiricidir
    grid = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        # ayirici mi?
        if all(set(c) <= set("-: ") and c for c in cells):
            continue
        grid.append(cells)
    if not grid:
        return
    ncol = max(len(r) for r in grid)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(grid):
        cells = t.add_row().cells
        for ci in range(ncol):
            txt = row[ci] if ci < len(row) else ""
            cell = cells[ci]
            cell.text = ""
            par = cell.paragraphs[0]
            add_runs_with_inline(par, txt)
            if ri == 0:
                for run in par.runs:
                    run.bold = True
    doc.add_paragraph()


def main():
    lines = MD.read_text(encoding="utf-8").split("\n")
    i = 0
    in_code = False
    code_buf = []
    table_buf = []
    cover_done = False  # kapak bloğunu (ilk --- öncesi) ortala

    def flush_code():
        if not code_buf:
            return
        p = doc.add_paragraph()
        r = p.add_run("\n".join(code_buf))
        r.font.name = "Consolas"; r.font.size = Pt(9)
        # gri arka plan yerine sadece monospace
        doc.add_paragraph()

    while i < len(lines):
        line = lines[i]

        # fenced code
        if line.strip().startswith("```"):
            if in_code:
                flush_code(); code_buf = []; in_code = False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue

        # tablo birikimi
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            table_buf.append(line); i += 1; continue
        elif table_buf:
            flush_table(table_buf); table_buf = []

        # gorsel ![alt](yol)
        m = re.match(r"!\[(.*?)\]\((.+?)\)", line.strip())
        if m:
            imgpath = (BASE / m.group(2)).resolve()
            if imgpath.exists():
                try:
                    doc.add_picture(str(imgpath), width=Inches(5.8))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e:
                    doc.add_paragraph(f"[Görsel yüklenemedi: {m.group(2)}]")
            else:
                doc.add_paragraph(f"[Görsel bulunamadı: {m.group(2)}]")
            i += 1; continue

        # basliklar
        if line.startswith("### "):
            h = doc.add_heading(level=3); add_runs_with_inline(h, line[4:].strip())
        elif line.startswith("## "):
            h = doc.add_heading(level=2); add_runs_with_inline(h, line[3:].strip())
        elif line.startswith("# "):
            h = doc.add_heading(level=1); add_runs_with_inline(h, line[2:].strip())
        elif line.strip() == "---":
            cover_done = True
            doc.add_paragraph()  # ayirici -> bos satir
        elif re.match(r"^\s*[-*]\s+", line):
            p = doc.add_paragraph(style="List Bullet")
            add_runs_with_inline(p, re.sub(r"^\s*[-*]\s+", "", line))
        elif re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            add_runs_with_inline(p, re.sub(r"^\s*\d+\.\s+", "", line))
        elif line.strip().startswith(">"):
            p = doc.add_paragraph()
            r0 = p.add_run(re.sub(r"^\s*>\s?", "", line))
            r0.italic = True; r0.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        elif line.strip() == "":
            pass
        else:
            p = doc.add_paragraph()
            add_runs_with_inline(p, line)
            # kapak bloğundaki satırları ortala ve başlığı büyüt
            if not cover_done:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                stripped = line.strip().replace("*", "")
                if "TÜRKÇE OTEL" in stripped:
                    for run in p.runs:
                        run.font.size = Pt(15)
                elif "İSTİNYE" in stripped:
                    for run in p.runs:
                        run.font.size = Pt(18)
        i += 1

    if table_buf:
        flush_table(table_buf)
    if code_buf:
        flush_code()

    doc.save(str(OUT))
    print("DOCX yazildi:", OUT)


if __name__ == "__main__":
    main()
