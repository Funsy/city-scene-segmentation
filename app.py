from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.config import get_device, load_config
from src.data import class_mask_to_rgb
from src.models import create_model
from src.utils import append_jsonl, ensure_dir


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    cfg = load_config(config_path)
    device = get_device(str(cfg.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(checkpoint["model_cfg"], num_classes=int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, cfg, device


def preprocess_rgb(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()


@torch.no_grad()
def run_prediction(model, checkpoint, device, image: Image.Image):
    image_rgb = np.array(image.convert("RGB"))
    original_size = image.size
    x = preprocess_rgb(image_rgb, int(checkpoint["image_size"])).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    conf_np = confidence.squeeze(0).cpu().numpy()
    palette = np.array(checkpoint["palette"], dtype=np.uint8)
    mask_rgb = class_mask_to_rgb(pred_np, palette, ignore_index=int(checkpoint.get("ignore_index", 255)))
    mask_rgb = cv2.resize(mask_rgb, original_size, interpolation=cv2.INTER_NEAREST)
    overlay = (0.55 * image_rgb + 0.45 * mask_rgb).clip(0, 255).astype(np.uint8)

    class_ids, counts = np.unique(pred_np, return_counts=True)
    total = counts.sum()
    rows = []
    for cls_id, count in zip(class_ids.tolist(), counts.tolist()):
        if cls_id < len(checkpoint["class_names"]):
            rows.append({
                "class_id": int(cls_id),
                "class_name": checkpoint["class_names"][cls_id],
                "pixel_share": float(count / total),
            })
    rows = sorted(rows, key=lambda x: x["pixel_share"], reverse=True)
    return mask_rgb, overlay, conf_np, elapsed, rows


st.set_page_config(page_title="City Scene Segmentation", layout="wide")
st.title("Семантическая сегментация городской сцены")
st.caption("Загрузка изображения, построение маски сегментации, расчёт уверенности и сохранение истории запусков.")

config_path = st.sidebar.text_input("Путь к config", "configs/camvid.yaml")
default_ckpt = "runs/unet_resnet18/best.pt"
checkpoint_path = st.sidebar.text_input("Путь к checkpoint", default_ckpt)

if not Path(checkpoint_path).exists():
    st.warning("Сначала обучи модель или укажи корректный путь к checkpoint .pt")
    st.code("python train.py --config configs/camvid.yaml --model unet_resnet18")
    st.stop()

model, checkpoint, cfg, device = load_model(checkpoint_path, config_path)
st.sidebar.write(f"Модель: **{checkpoint['model_name']}**")
st.sidebar.write(f"Устройство: **{device}**")
st.sidebar.write(f"Классов: **{checkpoint['num_classes']}**")

uploaded = st.file_uploader("Загрузи изображение городской сцены", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    mask_rgb, overlay, confidence, elapsed, rows = run_prediction(model, checkpoint, device, image)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Маска")
        st.image(mask_rgb, use_container_width=True)
    with col3:
        st.subheader("Наложение")
        st.image(overlay, use_container_width=True)

    st.metric("Средняя уверенность по пикселям", f"{float(confidence.mean()):.3f}")
    st.metric("Время обработки", f"{elapsed:.4f} сек")

    st.subheader("Краткая статистика по классам")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pixel_share"] = df["pixel_share"].map(lambda x: round(x, 4))
        st.dataframe(df, use_container_width=True)

    pred_dir = ensure_dir("runs/predictions")
    stem = Path(uploaded.name).stem
    mask_path = pred_dir / f"{stem}_{checkpoint['model_name']}_mask.png"
    overlay_path = pred_dir / f"{stem}_{checkpoint['model_name']}_overlay.png"
    Image.fromarray(mask_rgb).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    result = {
        "model_name": checkpoint["model_name"],
        "uploaded_name": uploaded.name,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "avg_confidence": float(confidence.mean()),
        "inference_time_sec": float(elapsed),
        "device": device,
        "classes_found": rows,
    }
    append_jsonl(result, cfg["outputs"].get("history_file", "runs/inference_history.jsonl"))

    st.success("Результат сохранён в runs/predictions, запись добавлена в историю запусков.")

st.divider()
st.subheader("История запусков")
history_path = Path(load_config(config_path)["outputs"].get("history_file", "runs/inference_history.jsonl"))
if history_path.exists():
    history_rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                history_rows.append(json.loads(line))
    if history_rows:
        st.dataframe(pd.DataFrame(history_rows).drop(columns=["classes_found"], errors="ignore"), use_container_width=True)
else:
    st.write("История пока пустая.")
