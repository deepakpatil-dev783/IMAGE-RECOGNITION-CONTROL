"""
reporting/report_generator.py
-------------------------------
Generates JSON + PDF quality reports from inspection results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, Image as RLImage,
                                    HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


class ReportGenerator:
    """Generates inspection reports in JSON and optionally PDF format."""

    def __init__(self, config: Dict):
        self.config = config
        self.reports_dir = Path(config["paths"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.company_name = config["reporting"].get("company_name", "Manufacturing Co.")

    # ── JSON ─────────────────────────────────────────────────────────────────

    def save_json(self, results: List[Dict], batch_id: str) -> str:
        summary = self._build_summary(results)
        report = {
            "report_id":    batch_id,
            "generated_at": datetime.now().isoformat(),
            "company":      self.company_name,
            "summary":      summary,
            "inspections":  results,
        }
        path = self.reports_dir / f"report_{batch_id}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report -> {path}")
        return str(path)

    # ── PDF ──────────────────────────────────────────────────────────────────

    def save_pdf(self, results: List[Dict], batch_id: str) -> Optional[str]:
        if not REPORTLAB_OK:
            print("⚠  reportlab not installed — skipping PDF generation.")
            return None

        path = self.reports_dir / f"report_{batch_id}.pdf"
        doc = SimpleDocTemplate(str(path), pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Title block ──
        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                     fontSize=22, alignment=TA_CENTER,
                                     textColor=colors.HexColor("#1a1a2e"))
        sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                   fontSize=10, alignment=TA_CENTER,
                                   textColor=colors.grey)

        story.append(Paragraph(f"{self.company_name}", title_style))
        story.append(Paragraph("Quality Control Inspection Report", sub_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#e94560")))
        story.append(Spacer(1, 0.5*cm))

        # ── Meta ──
        summary = self._build_summary(results)
        meta_data = [
            ["Report ID",     batch_id],
            ["Generated",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Total Inspected", str(summary["total"])],
            ["Passed",        str(summary["passed"])],
            ["Defects Found", str(summary["defects"])],
            ["Pass Rate",     f"{summary['pass_rate']:.1f}%"],
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 10*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (0, -1), colors.white),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (1, 0), (-1, -1),
             [colors.HexColor("#f0f4f8"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 1*cm))

        # ── Results table ──
        story.append(Paragraph("Inspection Results", styles["Heading2"]))
        story.append(Spacer(1, 0.3*cm))

        headers = ["#", "Image", "Verdict", "Confidence", "Defect Code", "Timestamp"]
        rows = [headers]
        for i, r in enumerate(results, 1):
            defect = r.get("defect_code") or {}
            code = defect.get("code", "—") if isinstance(defect, dict) else "—"
            verdict_color = colors.HexColor("#27ae60") if r["verdict"] == "PASS" else colors.HexColor("#e74c3c")
            rows.append([
                str(i),
                Path(r["image_path"]).name[:30],
                r.get("verdict", "—"),
                f"{r.get('confidence', 0):.1%}",
                code,
                r.get("timestamp", "")[:19],
            ])

        col_widths = [1*cm, 5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm]
        result_table = Table(rows, colWidths=col_widths, repeatRows=1)
        ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f9f9f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ])
        # Colour verdict cells
        for i, r in enumerate(results, 1):
            c = colors.HexColor("#d5f5e3") if r["verdict"] == "PASS" else colors.HexColor("#fadbd8")
            ts.add("BACKGROUND", (2, i), (2, i), c)
        result_table.setStyle(ts)
        story.append(result_table)

        # ── Footer ──
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                      fontSize=8, textColor=colors.grey,
                                      alignment=TA_CENTER)
        story.append(Paragraph(
            f"Generated by AI Quality Control System · {self.company_name} · CONFIDENTIAL",
            footer_style
        ))

        doc.build(story)
        print(f"PDF report -> {path}")
        return str(path)

    # ── Batch report entry point ───────────────────────────────────────────────

    def generate(self, results: List[Dict]) -> Dict[str, str]:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {
            "json": self.save_json(results, batch_id),
        }
        if self.config["reporting"].get("report_format", "pdf") == "pdf":
            pdf_path = self.save_pdf(results, batch_id)
            if pdf_path:
                paths["pdf"] = pdf_path
        return paths

    # ── Summary helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(results: List[Dict]) -> Dict:
        total   = len(results)
        passed  = sum(1 for r in results if r.get("verdict") == "PASS")
        defects = sum(1 for r in results if r.get("verdict") == "DEFECT")
        errors  = total - passed - defects
        defect_codes = {}
        for r in results:
            dc = r.get("defect_code")
            if dc and isinstance(dc, dict):
                code = dc["code"]
                defect_codes[code] = defect_codes.get(code, 0) + 1
        return {
            "total":       total,
            "passed":      passed,
            "defects":     defects,
            "errors":      errors,
            "pass_rate":   (passed / total * 100) if total else 0,
            "defect_rate": (defects / total * 100) if total else 0,
            "defect_breakdown": defect_codes,
        }
