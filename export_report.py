from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import load_config
from src.utils import ensure_dir


def setup_pdf_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("PracticeFont", path))
            return "PracticeFont"
    return "Helvetica"


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def export_excel(cfg: dict, out_path: Path) -> None:
    run_dir = Path(cfg["outputs"]["run_dir"])
    summary_path = run_dir / "summary.csv"
    history_path = Path(cfg["outputs"].get("history_file", "runs/inference_history.jsonl"))
    ensure_dir(out_path.parent)

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    inference_history = load_jsonl(history_path).drop(columns=["classes_found"], errors="ignore")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="models_summary", index=False)
        inference_history.to_excel(writer, sheet_name="inference_history", index=False)
        for model_dir in sorted(run_dir.glob("*/")):
            hist = model_dir / "history.csv"
            if hist.exists():
                sheet = model_dir.name[:31]
                pd.read_csv(hist).to_excel(writer, sheet_name=sheet, index=False)


def export_pdf(cfg: dict, out_path: Path) -> None:
    run_dir = Path(cfg["outputs"]["run_dir"])
    summary_path = run_dir / "summary.csv"
    ensure_dir(out_path.parent)
    font_name = setup_pdf_font()
    styles = getSampleStyleSheet()
    styles["Title"].fontName = font_name
    styles["BodyText"].fontName = font_name
    doc = SimpleDocTemplate(str(out_path), pagesize=landscape(A4), rightMargin=1 * cm, leftMargin=1 * cm, topMargin=1 * cm, bottomMargin=1 * cm)
    story = []
    story.append(Paragraph("Краткий отчёт по сравнению моделей семантической сегментации", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    if not summary_path.exists():
        story.append(Paragraph("Файл runs/summary.csv не найден. Сначала выполните обучение и оценку моделей.", styles["BodyText"]))
        doc.build(story)
        return

    summary = pd.read_csv(summary_path)
    cols = [
        "model_name",
        "split",
        "mean_iou",
        "mean_dice",
        "pixel_accuracy",
        "fps",
        "avg_inference_time_sec",
        "checkpoint_mb",
        "device",
    ]
    available_cols = [c for c in cols if c in summary.columns]
    table_df = summary[available_cols].copy()
    for col in ["mean_iou", "mean_dice", "pixel_accuracy", "fps", "avg_inference_time_sec", "checkpoint_mb"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

    data = [available_cols] + table_df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTNAME", (0, 0), (-1, 0), font_name),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Вывод для отчёта: лучшую модель выбирайте не только по mean IoU, но и по скорости, размеру checkpoint и характеру ошибок на примерах.", styles["BodyText"]))
    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model comparison report to Excel and PDF.")
    parser.add_argument("--config", default="configs/camvid.yaml")
    parser.add_argument("--excel", default="report/model_comparison.xlsx")
    parser.add_argument("--pdf", default="report/model_comparison.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    export_excel(cfg, Path(args.excel))
    export_pdf(cfg, Path(args.pdf))
    print(f"Saved: {args.excel}")
    print(f"Saved: {args.pdf}")
