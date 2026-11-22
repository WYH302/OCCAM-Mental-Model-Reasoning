#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a publication-grade, 1:1 synchronized Vector Framework Architecture Diagram:
  1. outputs/paper_assets/fig0_framework_architecture.pdf  (Type-42 Comic Sans MS Vector PDF)
  2. outputs/paper_assets/fig0_framework_architecture.png  (300 DPI Raster Preview)
  3. outputs/paper_assets/fig0_framework_architecture.svg  (Clean Editable Vector SVG)
  4. outputs/paper_assets/OCCAM_Framework_Architecture.pptx (1:1 Editable 1-Slide PowerPoint)

Uses native Windows Comic Sans MS (C:/Windows/Fonts/comic.ttf & comicbd.ttf) on a locked
12.00 x 5.25 inch canvas with deep color-coded headers, clean typography, and zero text overflow.
"""
import html
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

for fpath in (r"C:\Windows\Fonts\comic.ttf", r"C:\Windows\Fonts\comicbd.ttf", r"C:\Windows\Fonts\comici.ttf"):
    if Path(fpath).exists():
        fm.fontManager.addfont(fpath)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "Comic Sans MS",
})

W_IN = 12.00
H_IN = 5.25


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class DiagramBuilder:
    """Single-source declarative canvas emitting synchronized Matplotlib (PDF/PNG), SVG, and PPTX."""

    def __init__(self, width_in: float = W_IN, height_in: float = H_IN):
        self.w = width_in
        self.h = height_in
        self.boxes: List[dict] = []
        self.arrows: List[dict] = []
        self.labels: List[dict] = []
        self.polylines: List[dict] = []

    def add_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        edge: str,
        lw: float = 1.2,
        rad: float = 0.08,
        linestyle: str = "solid",
        shadow: bool = False,
        title: str = "",
        body: str = "",
        title_color: str = "#0F172A",
        body_color: str = "#334155",
        title_pt: float = 8.4,
        body_pt: float = 7.2,
        align: str = "center",
        zorder: int = 2,
    ):
        self.boxes.append({
            "x": x, "y": y, "w": w, "h": h,
            "fill": fill, "edge": edge, "lw": lw, "rad": rad,
            "linestyle": linestyle, "shadow": shadow,
            "title": title, "body": body,
            "title_color": title_color, "body_color": body_color,
            "title_pt": title_pt, "body_pt": body_pt,
            "align": align, "zorder": zorder,
        })

    def add_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#334155",
        lw: float = 1.6,
        style: str = "-|>",
        rad_arc: float = 0.0,
        zorder: int = 6,
    ):
        self.arrows.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "color": color, "lw": lw, "style": style, "rad_arc": rad_arc, "zorder": zorder,
        })

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        pt: float = 8.0,
        color: str = "#0F172A",
        bold: bool = False,
        italic: bool = False,
        ha: str = "center",
        va: str = "center",
        zorder: int = 8,
    ):
        self.labels.append({
            "x": x, "y": y, "text": text, "pt": pt, "color": color,
            "bold": bold, "italic": italic, "ha": ha, "va": va, "zorder": zorder,
        })

    def add_scribble_loop(self, cx: float, cy: float, rx: float, ry: float, color: str = "#334155", lw: float = 1.1, zorder: int = 5):
        """Clean mathematical Lissajous-inspired tangle representing Long-CoT overthinking loops."""
        t = np.linspace(0, 2 * np.pi, 180)
        xs = cx + rx * (0.62 * np.sin(3 * t + 0.4) + 0.38 * np.cos(5 * t))
        ys = cy + ry * (0.60 * np.cos(2 * t + 0.2) + 0.40 * np.sin(7 * t))
        self.polylines.append({"xs": xs.tolist(), "ys": ys.tolist(), "color": color, "lw": lw, "zorder": zorder})

    def _compute_text_centers(self, b: dict) -> Tuple[float, float]:
        """Balanced vertical layout for title + multiline body without excessive white gap."""
        n_t = b["title"].count("\n") + 1
        n_b = b["body"].count("\n") + 1
        h_t = n_t * (b["title_pt"] / 72.0) * 1.28
        h_b = n_b * (b["body_pt"] / 72.0) * 1.30
        gap = 0.055
        tot_h = h_t + gap + h_b
        y_mid = b["y"] + b["h"] / 2.0
        y_top = y_mid + tot_h / 2.0
        y_title = y_top - h_t / 2.0
        y_body = y_top - h_t - gap - h_b / 2.0
        return y_title, y_body

    def render_pdf_png(self, pdf_path: Path, png_path: Path):
        fig = plt.figure(figsize=(self.w, self.h), dpi=300, facecolor="white")
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0.0, self.w)
        ax.set_ylim(0.0, self.h)
        ax.axis("off")

        for b in self.boxes:
            if b["shadow"]:
                sh = FancyBboxPatch(
                    (b["x"] + 0.032, b["y"] - 0.032), b["w"], b["h"],
                    boxstyle=f"round,pad=0.0,rounding_size={b['rad']}",
                    facecolor="#94A3B8", edgecolor="none", alpha=0.28, zorder=b["zorder"] - 0.5,
                )
                ax.add_patch(sh)
            ls = "--" if b["linestyle"] == "dashed" else "-"
            patch = FancyBboxPatch(
                (b["x"], b["y"]), b["w"], b["h"],
                boxstyle=f"round,pad=0.0,rounding_size={b['rad']}",
                facecolor=b["fill"], edgecolor=b["edge"],
                linewidth=b["lw"], linestyle=ls, zorder=b["zorder"],
            )
            ax.add_patch(patch)

            cx = b["x"] + b["w"] / 2.0 if b["align"] == "center" else b["x"] + 0.11
            ha = b["align"]
            if b["title"] and b["body"]:
                y_title, y_body = self._compute_text_centers(b)
                ax.text(
                    cx, y_title, b["title"],
                    fontsize=b["title_pt"], fontweight="bold", color=b["title_color"],
                    ha=ha, va="center", linespacing=1.15, fontfamily="Comic Sans MS", zorder=b["zorder"] + 2,
                )
                ax.text(
                    cx, y_body, b["body"],
                    fontsize=b["body_pt"], fontweight="normal", color=b["body_color"],
                    ha=ha, va="center", linespacing=1.22, fontfamily="Comic Sans MS", zorder=b["zorder"] + 2,
                )
            elif b["title"]:
                ax.text(
                    cx, b["y"] + b["h"] / 2.0, b["title"],
                    fontsize=b["title_pt"], fontweight="bold", color=b["title_color"],
                    ha=ha, va="center", linespacing=1.16, fontfamily="Comic Sans MS", zorder=b["zorder"] + 2,
                )
            elif b["body"]:
                ax.text(
                    cx, b["y"] + b["h"] / 2.0, b["body"],
                    fontsize=b["body_pt"], fontweight="normal", color=b["body_color"],
                    ha=ha, va="center", linespacing=1.20, fontfamily="Comic Sans MS", zorder=b["zorder"] + 2,
                )

        for pl in self.polylines:
            ax.plot(pl["xs"], pl["ys"], color=pl["color"], linewidth=pl["lw"], alpha=0.82, zorder=pl["zorder"])

        for a in self.arrows:
            conn = f"arc3,rad={a['rad_arc']}" if abs(a["rad_arc"]) > 1e-4 else "arc3,rad=0.0"
            arr = FancyArrowPatch(
                (a["x1"], a["y1"]), (a["x2"], a["y2"]),
                arrowstyle=f"{a['style']},head_length=5.8,head_width=3.8",
                connectionstyle=conn,
                color=a["color"], linewidth=a["lw"], zorder=a["zorder"],
            )
            ax.add_patch(arr)

        for lb in self.labels:
            ax.text(
                lb["x"], lb["y"], lb["text"],
                fontsize=lb["pt"],
                fontweight="bold" if lb["bold"] else "normal",
                fontstyle="italic" if lb["italic"] else "normal",
                color=lb["color"], ha=lb["ha"], va=lb["va"],
                fontfamily="Comic Sans MS", zorder=lb["zorder"],
            )

        fig.savefig(pdf_path, facecolor="white")
        fig.savefig(png_path, dpi=300, facecolor="white")
        plt.close(fig)

    def render_svg(self, svg_path: Path):
        S = 96.0
        W_px, H_px = self.w * S, self.h * S

        def sy(y_in: float) -> float:
            return (self.h - y_in) * S

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_px:.1f}" height="{H_px:.1f}" viewBox="0 0 {W_px:.1f} {H_px:.1f}">',
            '  <defs>',
            '    <filter id="cardShadow" x="-4%" y="-4%" width="110%" height="112%">',
            '      <feDropShadow dx="2" dy="2" stdDeviation="2.5" flood-color="#94A3B8" flood-opacity="0.28"/>',
            '    </filter>',
        ]
        arrow_colors = sorted({a["color"] for a in self.arrows})
        for idx_c, col in enumerate(arrow_colors):
            cid = f"arr_{idx_c}"
            lines.append(
                f'    <marker id="{cid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                f'<path d="M 0 1 L 10 5 L 0 9 z" fill="{col}"/></marker>'
            )
        col_to_id = {c: f"arr_{i}" for i, c in enumerate(arrow_colors)}
        lines.append('  </defs>')
        lines.append(f'  <rect x="0" y="0" width="{W_px:.1f}" height="{H_px:.1f}" fill="#FFFFFF"/>')

        for b in sorted(self.boxes, key=lambda d: d["zorder"]):
            rx, ry = b["x"] * S, sy(b["y"] + b["h"])
            rw, rh, rr = b["w"] * S, b["h"] * S, b["rad"] * S
            dash = ' stroke-dasharray="6,4"' if b["linestyle"] == "dashed" else ""
            filt = ' filter="url(#cardShadow)"' if b["shadow"] else ""
            lines.append(
                f'  <rect x="{rx:.2f}" y="{ry:.2f}" width="{rw:.2f}" height="{rh:.2f}" '
                f'rx="{rr:.2f}" ry="{rr:.2f}" fill="{b["fill"]}" stroke="{b["edge"]}" '
                f'stroke-width="{b["lw"] * 1.15:.2f}"{dash}{filt}/>'
            )

            anchor = "middle" if b["align"] == "center" else "start"
            tx = (b["x"] + b["w"] / 2.0) * S if b["align"] == "center" else (b["x"] + 0.11) * S

            def emit_ml(txt_s: str, yc_in: float, pt_sz: float, col_s: str, wt: str):
                parts = txt_s.split("\n")
                px_sz = pt_sz * 1.333
                lh = px_sz * 1.24
                y0_px = sy(yc_in) - (len(parts) - 1) * lh / 2.0 + px_sz * 0.34
                for pi, seg in enumerate(parts):
                    lines.append(
                        f'  <text x="{tx:.2f}" y="{y0_px + pi * lh:.2f}" font-family="Comic Sans MS, Comic Neue, cursive, sans-serif" '
                        f'font-size="{px_sz:.2f}" font-weight="{wt}" fill="{col_s}" text-anchor="{anchor}">{html.escape(seg)}</text>'
                    )

            if b["title"] and b["body"]:
                yt, yb = self._compute_text_centers(b)
                emit_ml(b["title"], yt, b["title_pt"], b["title_color"], "bold")
                emit_ml(b["body"], yb, b["body_pt"], b["body_color"], "normal")
            elif b["title"]:
                emit_ml(b["title"], b["y"] + b["h"] / 2.0, b["title_pt"], b["title_color"], "bold")
            elif b["body"]:
                emit_ml(b["body"], b["y"] + b["h"] / 2.0, b["body_pt"], b["body_color"], "normal")

        for pl in self.polylines:
            pts = " ".join(f"{x * S:.1f},{sy(y):.1f}" for x, y in zip(pl["xs"], pl["ys"]))
            lines.append(f'  <polyline points="{pts}" fill="none" stroke="{pl["color"]}" stroke-width="{pl["lw"] * 1.15:.2f}" stroke-opacity="0.82"/>')

        for a in self.arrows:
            x1, y1 = a["x1"] * S, sy(a["y1"])
            x2, y2 = a["x2"] * S, sy(a["y2"])
            cid = col_to_id[a["color"]]
            if abs(a["rad_arc"]) > 1e-4:
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dx, dy = x2 - x1, y2 - y1
                cx, cy = mx - a["rad_arc"] * dy, my + a["rad_arc"] * dx
                lines.append(
                    f'  <path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" '
                    f'fill="none" stroke="{a["color"]}" stroke-width="{a["lw"] * 1.25:.2f}" marker-end="url(#{cid})"/>'
                )
            else:
                lines.append(
                    f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                    f'stroke="{a["color"]}" stroke-width="{a["lw"] * 1.25:.2f}" marker-end="url(#{cid})"/>'
                )

        for lb in self.labels:
            tx, ty = lb["x"] * S, sy(lb["y"])
            px_sz = lb["pt"] * 1.333
            anc = {"center": "middle", "left": "start", "right": "end"}[lb["ha"]]
            wt = "bold" if lb["bold"] else "normal"
            st = "italic" if lb["italic"] else "normal"
            lines.append(
                f'  <text x="{tx:.2f}" y="{ty + px_sz * 0.34:.2f}" font-family="Comic Sans MS, Comic Neue, cursive, sans-serif" '
                f'font-size="{px_sz:.2f}" font-weight="{wt}" font-style="{st}" fill="{lb["color"]}" text-anchor="{anc}">{html.escape(lb["text"])}</text>'
            )

        lines.append("</svg>")
        svg_path.write_text("\n".join(lines), encoding="utf-8")

    def render_pptx(self, pptx_path: Path):
        prs = Presentation()
        prs.slide_width = Inches(self.w)
        prs.slide_height = Inches(self.h)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = RGBColor(255, 255, 255)

        for b in sorted(self.boxes, key=lambda d: d["zorder"]):
            top_in = self.h - (b["y"] + b["h"])
            shp = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(b["x"]), Inches(top_in), Inches(b["w"]), Inches(b["h"])
            )
            shp.adjustments[0] = min(0.24, b["rad"] / max(0.2, min(b["w"], b["h"])))
            shp.fill.solid()
            shp.fill.fore_color.rgb = RGBColor(*hex_to_rgb(b["fill"]))
            shp.line.color.rgb = RGBColor(*hex_to_rgb(b["edge"]))
            shp.line.width = Pt(b["lw"] * 0.95)
            if b["linestyle"] == "dashed":
                shp.line.dash_style = MSO_LINE.DASH

            tf = shp.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = Inches(0.08)
            tf.margin_right = Inches(0.08)
            tf.margin_top = Inches(0.04)
            tf.margin_bottom = Inches(0.04)

            align_p = PP_ALIGN.CENTER if b["align"] == "center" else PP_ALIGN.LEFT
            first_used = False
            if b["title"]:
                for ti, tseg in enumerate(b["title"].split("\n")):
                    p = tf.paragraphs[0] if ti == 0 else tf.add_paragraph()
                    p.alignment = align_p
                    run = p.add_run()
                    run.text = tseg
                    run.font.name = "Comic Sans MS"
                    run.font.size = Pt(b["title_pt"])
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(*hex_to_rgb(b["title_color"]))
                first_used = True
            if b["body"]:
                for bi, line_txt in enumerate(b["body"].split("\n")):
                    p = tf.paragraphs[0] if (not first_used and bi == 0) else tf.add_paragraph()
                    p.alignment = align_p
                    p.space_before = Pt(2.0 if bi == 0 and first_used else 0.5)
                    run = p.add_run()
                    run.text = line_txt
                    run.font.name = "Comic Sans MS"
                    run.font.size = Pt(b["body_pt"])
                    run.font.bold = False
                    run.font.color.rgb = RGBColor(*hex_to_rgb(b["body_color"]))

        for a in self.arrows:
            x1, y1 = Inches(a["x1"]), Inches(self.h - a["y1"])
            x2, y2 = Inches(a["x2"]), Inches(self.h - a["y2"])
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
            conn.line.color.rgb = RGBColor(*hex_to_rgb(a["color"]))
            conn.line.width = Pt(a["lw"] * 1.05)
            ln_el = conn.line._get_or_add_ln()
            tail = OxmlElement("a:tailEnd")
            tail.set("type", "triangle")
            tail.set("w", "med")
            tail.set("len", "med")
            ln_el.append(tail)

        for lb in self.labels:
            tb_w = 3.8 if len(lb["text"]) > 30 else 1.2
            tb_h = 0.28
            left_in = lb["x"] - (tb_w / 2.0 if lb["ha"] == "center" else 0.0)
            top_in = self.h - lb["y"] - tb_h / 2.0
            tb = slide.shapes.add_textbox(Inches(left_in), Inches(top_in), Inches(tb_w), Inches(tb_h))
            tf = tb.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT}[lb["ha"]]
            run = p.add_run()
            run.text = lb["text"]
            run.font.name = "Comic Sans MS"
            run.font.size = Pt(lb["pt"])
            run.font.bold = lb["bold"]
            run.font.italic = lb["italic"]
            run.font.color.rgb = RGBColor(*hex_to_rgb(lb["color"]))

        prs.save(str(pptx_path))


def build_occam_architecture() -> DiagramBuilder:
    db = DiagramBuilder(W_IN, H_IN)

    # Clean Title Strip
    db.add_text(
        6.00, 5.02,
        "OCCAM: Regime-Adaptive Mental-Model Cognitive Compression for Language Models",
        pt=11.4, color="#0F172A", bold=True, zorder=5,
    )

    # =========================================================================
    # PANEL (a): PROBLEM MOTIVATION (x = 0.12 .. 3.22, w = 3.10)
    # =========================================================================
    db.add_box(0.12, 0.14, 3.10, 4.66, fill="#FFF1F2", edge="#F43F5E", lw=1.35, rad=0.11, shadow=True, zorder=1)
    db.add_box(
        0.12, 4.32, 3.10, 0.48,
        fill="#E11D48", edge="#E11D48", lw=1.0, rad=0.09,
        title="(a) Bottleneck: Long-CoT Overthinking",
        title_color="#FFFFFF", title_pt=8.8, zorder=3,
    )
    # Input Query Card
    db.add_box(
        0.46, 3.74, 2.42, 0.44,
        fill="#FFFFFF", edge="#FDA4AF", lw=1.0, rad=0.07,
        title="Input Query x",
        body="Causal / Deductive / Math Problem",
        title_pt=8.0, body_pt=6.8, zorder=3,
    )
    db.add_arrow(1.67, 3.74, 1.67, 3.52, color="#E11D48", lw=1.55)

    # Tangle Overthinking Card
    db.add_box(
        0.26, 2.04, 2.82, 1.46,
        fill="#FFFFFF", edge="#FB7185", lw=1.2, rad=0.08, linestyle="dashed",
        title="", body="", zorder=3,
    )
    db.add_text(1.67, 3.32, '"Wait, let me re-check..." (1,458 tokens)', pt=7.6, color="#BE123C", bold=True, zorder=5)
    db.add_scribble_loop(1.67, 2.74, 0.72, 0.36, color="#1E293B", lw=1.15, zorder=5)
    db.add_text(1.67, 2.22, "Repetitive self-doubt loops (p95 = 5,821t)", pt=6.9, color="#475569", italic=True, zorder=5)

    # Two solid Crimson diagnostic pills
    db.add_arrow(1.02, 2.04, 1.02, 1.80, color="#E11D48", lw=1.45)
    db.add_arrow(2.32, 2.04, 2.32, 1.80, color="#E11D48", lw=1.45)
    db.add_box(
        0.26, 1.12, 1.36, 0.66,
        fill="#E11D48", edge="#BE123C", lw=0.9, rad=0.07,
        title="Heavy KV-Cache",
        body="79.7 MB / query",
        title_color="#FFFFFF", body_color="#FFE4E6", title_pt=7.7, body_pt=6.8, zorder=3,
    )
    db.add_box(
        1.72, 1.12, 1.36, 0.66,
        fill="#E11D48", edge="#BE123C", lw=0.9, rad=0.07,
        title="Framing Collapse",
        body="-12.6% ARC-C",
        title_color="#FFFFFF", body_color="#FFE4E6", title_pt=7.7, body_pt=6.8, zorder=3,
    )

    # Rate-Distortion Diagnosis Card
    db.add_box(
        0.26, 0.26, 2.82, 0.74,
        fill="#FFFFFF", edge="#FDA4AF", lw=1.0, rad=0.07,
        title="Rate-Distortion Dichotomy (Prop. 1)",
        body="Naive 200t budget cap crashes R1 (-30.9%).\nCompression must adapt to task regime!",
        title_color="#9F1239", body_color="#334155", title_pt=7.6, body_pt=6.7, zorder=3,
    )

    # =========================================================================
    # PANEL (b): COMPILE-TIME INTERNALIZATION (x = 3.54 .. 7.72, w = 4.18)
    # =========================================================================
    db.add_box(3.54, 0.14, 4.18, 4.66, fill="#EFF6FF", edge="#3B82F6", lw=1.35, rad=0.11, shadow=True, zorder=1)
    db.add_box(
        3.54, 4.32, 4.18, 0.48,
        fill="#1D4ED8", edge="#1D4ED8", lw=1.0, rad=0.09,
        title="(b) Compile-Time Internalization (Single 24 GB GPU)",
        title_color="#FFFFFF", title_pt=8.6, zorder=3,
    )

    # B1: Canonical 16-Primitive Basis
    db.add_box(3.68, 2.98, 3.90, 1.24, fill="#FFFFFF", edge="#93C5FD", lw=1.05, rad=0.08, shadow=True, zorder=2)
    db.add_text(
        5.63, 4.08,
        "Canonical Basis M16 (Nobel/Turing & PFC-Hippocampal Brain Taxonomy)",
        pt=7.25, color="#1E40AF", bold=True, zorder=4,
    )
    chips = [
        (3.78, 3.54, 1.81, 0.42, "#E0F2FE", "#0284C7", "D1 Meta-Control (rlPFC/dACC)", "Inversion, Isomorphism, Backtrack"),
        (5.67, 3.54, 1.81, 0.42, "#EDE9FE", "#7C3AED", "D2 Invariants (Friston MDL/IPS)", "Axioms, MDL Razor, Falsification"),
        (3.78, 3.06, 1.81, 0.42, "#CCFBF1", "#0D9488", "D3 Representations (OFC-Hipp.)", "Causal DAG, Min-Cut, Graph Rank"),
        (5.67, 3.06, 1.81, 0.42, "#FEF3C7", "#D97706", "D4 Strategic Ops (mPFC-TPJ)", "Bayes, Blotto/Nash, Convexity"),
    ]
    for cx, cy, cw, ch, cf, ce, ct, cb in chips:
        db.add_box(cx, cy, cw, ch, fill=cf, edge=ce, lw=0.85, rad=0.05,
                   title=ct, body=cb, title_color="#0F172A", body_color="#334155",
                   title_pt=6.65, body_pt=6.05, zorder=4)

    db.add_arrow(5.63, 2.98, 5.63, 2.76, color="#1D4ED8", lw=1.65)

    # B2: Local Self-Distillation & SymPy Gate
    db.add_box(3.68, 1.72, 3.90, 1.02, fill="#FFFFFF", edge="#93C5FD", lw=1.05, rad=0.08, shadow=True, zorder=2)
    db.add_text(
        5.63, 2.60,
        "Stage 1: Local Self-Distillation & SymPy Gate (OCCAM-2K, $0 API)",
        pt=7.45, color="#1E3A8A", bold=True, zorder=4,
    )
    db.add_box(
        3.78, 1.82, 1.74, 0.62,
        fill="#CCFBF1", edge="#0F766E", lw=0.95, rad=0.06,
        title="Verified Chosen y+ (233t)",
        body="<cog_router> + <cog_trace>\nRegime Tag • 0.00% Overlap",
        title_color="#0F766E", body_color="#134E4A", title_pt=7.0, body_pt=6.2, zorder=4,
    )
    db.add_box(
        5.74, 1.82, 1.74, 0.62,
        fill="#FEF3C7", edge="#D97706", lw=0.95, rad=0.06,
        title="Verbose Negative y- (544t)",
        body="Unstructured CoT Rollouts\n1,570 Verified Pairs",
        title_color="#B45309", body_color="#78350F", title_pt=7.0, body_pt=6.2, zorder=4,
    )
    db.add_text(5.63, 2.13, "vs.", pt=7.3, color="#475569", bold=True, zorder=5)

    db.add_arrow(5.63, 1.72, 5.63, 1.50, color="#1D4ED8", lw=1.65)

    # B3: Strip-Prompt Anchored LoRA + OCCAM-SimPO
    db.add_box(
        3.68, 0.26, 3.90, 1.22,
        fill="#FFFFFF", edge="#1D4ED8", lw=1.35, rad=0.08, shadow=True,
        title="Stage 2: Strip-Prompt Anchored LoRA + OCCAM-SimPO",
        body="• Strip M16 prompt -> Native <think>-aligned SFT (0 prompt cost)\n"
             "• Router credit weighting (alpha_r > 1 on <cog_router> span)\n"
             "• Dynamic penalty lambda_len(Z) + calibrated horizon K = 25\n"
             "• 256-tok chunked lm_head: <=17.8 GB peak bf16 VRAM",
        title_color="#1D4ED8", body_color="#1E293B", title_pt=7.7, body_pt=6.65,
        align="left", zorder=3,
    )

    # =========================================================================
    # PANEL (c): ZERO-PROMPT REGIME-ADAPTIVE INFERENCE (x = 8.04 .. 11.88, w = 3.84)
    # =========================================================================
    db.add_box(8.04, 0.14, 3.84, 4.66, fill="#F0FDFA", edge="#14B8A6", lw=1.35, rad=0.11, shadow=True, zorder=1)
    db.add_box(
        8.04, 4.32, 3.84, 0.48,
        fill="#0F766E", edge="#0F766E", lw=1.0, rad=0.09,
        title="(c) Zero-Prompt Regime-Adaptive Inference",
        title_color="#FFFFFF", title_pt=8.6, zorder=3,
    )

    # Bare test query
    db.add_box(
        8.46, 3.74, 3.00, 0.44,
        fill="#FFFFFF", edge="#5EEAD4", lw=1.0, rad=0.07,
        title="Bare Test Query x (Zero System Prompt)",
        body="No External Schema • No Benchmark Tag",
        title_pt=7.75, body_pt=6.65, zorder=3,
    )
    db.add_arrow(9.96, 3.74, 9.96, 3.52, color="#0F766E", lw=1.55)

    # Phase 1 Router Switch
    db.add_box(
        8.18, 2.66, 3.56, 0.84,
        fill="#FFFFFF", edge="#0F766E", lw=1.35, rad=0.08, shadow=True,
        title="Phase 1: Internalized <cog_router> (<12 bits, ~55t)",
        body="[Topology] • [Bias Pruned] • [Primitives P_i -> P_j]\n"
             "Autoregressive [Execution_Regime] Gate (1 GPU Engine)",
        title_color="#0F766E", body_color="#1E293B", title_pt=7.65, body_pt=6.65, zorder=3,
    )

    # Clean split arrows
    db.add_arrow(9.18, 2.66, 9.04, 2.34, color="#0F766E", lw=1.7)
    db.add_arrow(10.74, 2.66, 10.88, 2.34, color="#D97706", lw=1.7)

    # Regime I Branch (Rich Deep Emerald-Teal Card matching high-contrast academic style!)
    db.add_box(
        8.18, 1.06, 1.72, 1.26,
        fill="#0F766E", edge="#115E59", lw=1.1, rad=0.08, shadow=True,
        title="Regime I:\nStructural Framing",
        body="OCCAM LoRA Active\nPrefix-Cached <cog_trace>\n~200t • -74% Entropy\nCures Overthinking",
        title_color="#FFFFFF", body_color="#CCFBF1", title_pt=7.4, body_pt=6.45, zorder=3,
    )

    # Regime II Branch (Rich Warm Amber Card!)
    db.add_box(
        10.02, 1.06, 1.72, 1.26,
        fill="#D97706", edge="#B45309", lw=1.1, rad=0.08, shadow=True,
        title="Regime II:\nSymbolic Rollout",
        body="Uncompressed Rollout\n(Base / R1 Specialist)\nRespects Fano Floor\n87.4% MATH-500",
        title_color="#FFFFFF", body_color="#FEF3C7", title_pt=7.4, body_pt=6.45, zorder=3,
    )

    # Merge arrows into Final Answer Banner
    db.add_arrow(9.04, 1.06, 9.54, 0.78, color="#0F766E", lw=1.7)
    db.add_arrow(10.88, 1.06, 10.38, 0.78, color="#D97706", lw=1.7)

    db.add_box(
        8.18, 0.26, 3.56, 0.50,
        fill="#0F172A", edge="#0F172A", lw=1.0, rad=0.08,
        title="Verified Final <answer>  (#1 SOTA on 7B-Active)",
        body="74.8% ID • 67.9% Fr4 • 76.4% All-8 • 2.1x-6.6x Fewer Tokens",
        title_color="#FFFFFF", body_color="#E2E8F0", title_pt=7.8, body_pt=6.7, zorder=4,
    )

    # Inter-Panel Connecting Arrows
    db.add_arrow(3.22, 2.48, 3.54, 2.48, color="#1E293B", lw=2.1, zorder=7)
    db.add_text(3.38, 2.70, "Solve", pt=7.1, color="#1E293B", bold=True, zorder=8)

    db.add_arrow(7.72, 2.48, 8.04, 2.48, color="#0F766E", lw=2.1, zorder=7)
    db.add_text(7.88, 2.70, "Deploy", pt=7.1, color="#0F766E", bold=True, zorder=8)

    return db


def main():
    db = build_occam_architecture()
    pdf_p = ASSETS_DIR / "fig0_framework_architecture.pdf"
    png_p = ASSETS_DIR / "fig0_framework_architecture.png"
    svg_p = ASSETS_DIR / "fig0_framework_architecture.svg"
    pptx_p = ASSETS_DIR / "OCCAM_Framework_Architecture.pptx"

    db.render_pdf_png(pdf_p, png_p)
    db.render_svg(svg_p)
    db.render_pptx(pptx_p)

    # Also copy the AI-generated Graphical Abstract render into paper_assets
    ai_ga = Path(r"C:\Users\user\.gemini\antigravity\brain\95852fc5-11c9-4cdb-9232-19406abfbeca\occam_framework_architecture_v3_1795267650901.jpg")
    if ai_ga.exists():
        import shutil
        shutil.copy2(ai_ga, ASSETS_DIR / "graphical_abstract_elsevier.jpg")

    print("[FRAMEWORK ARCHITECTURE v2 BUILT 1:1 IN COMIC SANS MS]")
    for p in (pdf_p, png_p, svg_p, pptx_p):
        print(f"  -> {p.name:<40} ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
