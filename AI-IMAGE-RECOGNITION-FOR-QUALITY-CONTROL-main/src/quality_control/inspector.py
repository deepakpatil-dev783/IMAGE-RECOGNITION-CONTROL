"""
quality_control/inspector.py
------------------------------
Runs inference on product images and classifies PASS / DEFECT.
"""

import json
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.model.cnn_model import QualityControlCNN, GradCAM, load_checkpoint
from src.preprocessing.image_processor import ImagePreprocessor


DEFECT_CODES = {
    "D001": "Surface Scratch",
    "D002": "Dent / Deformation",
    "D003": "Color Mismatch",
    "D004": "Missing Component",
    "D005": "Dimensional Deviation",
}

CLASS_NAMES = {0: "PASS", 1: "DEFECT"}


class QualityInspector:
    """
    End-to-end quality inspector:
      1. Preprocess image
      2. Run CNN inference
      3. Generate Grad-CAM heatmap
      4. Return structured result dict
    """

    def __init__(self, model_path: str, config: Dict, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = config["inference"]["confidence_threshold"]
        self.preprocessor = ImagePreprocessor(config["model"]["input_size"])

        # Load model
        self.model = QualityControlCNN(
            num_classes=config["model"]["num_classes"],
            pretrained=False,
            dropout=config["model"]["dropout"],
        ).to(self.device)
        load_checkpoint(self.model, model_path, self.device)
        self.model.eval()

        self.grad_cam = GradCAM(self.model)

    # ── Single image ──────────────────────────────────────────────────────────

    def inspect(self, image_path: str) -> Dict:
        tensor, original = self.preprocessor.preprocess(image_path)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        defect_prob = probs[1].item()
        pass_prob   = probs[0].item()
        pred_class  = 1 if defect_prob >= self.threshold else 0
        verdict     = CLASS_NAMES[pred_class]

        # Grad-CAM for defects
        heatmap_path = None
        if pred_class == 1:
            cam = self.grad_cam.generate(tensor, class_idx=1).cpu().numpy()
            heatmap_path = self._overlay_heatmap(original, cam, image_path)

        result = {
            "inspection_id":   str(uuid.uuid4())[:8],
            "timestamp":       datetime.now().isoformat(),
            "image_path":      str(image_path),
            "verdict":         verdict,
            "pass_probability":   round(pass_prob, 4),
            "defect_probability": round(defect_prob, 4),
            "confidence":      round(max(pass_prob, defect_prob), 4),
            "heatmap_path":    heatmap_path,
            "defect_code":     self._assign_defect_code(defect_prob) if pred_class == 1 else None,
        }
        return result

    # ── Batch ─────────────────────────────────────────────────────────────────

    def inspect_batch(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for path in image_paths:
            try:
                results.append(self.inspect(path))
            except Exception as e:
                results.append({
                    "image_path": path,
                    "verdict": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
        return results

    def inspect_folder(self, folder: str) -> List[Dict]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        paths = [str(p) for p in Path(folder).iterdir()
                 if p.suffix.lower() in extensions]
        print(f"Found {len(paths)} images in {folder}")
        return self.inspect_batch(paths)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _overlay_heatmap(self, original: np.ndarray,
                         cam: np.ndarray,
                         source_path: str) -> str:
        h, w = original.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR), 0.6,
            heatmap, 0.4, 0
        )
        out_dir = Path("data/reports/heatmaps")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source_path).stem
        out_path = str(out_dir / f"{stem}_heatmap.jpg")
        cv2.imwrite(out_path, overlay)
        return out_path

    def _assign_defect_code(self, defect_prob: float) -> Dict:
        """Placeholder: in production, use a multi-label head or rule-based system."""
        if defect_prob > 0.95:
            code = "D001"
        elif defect_prob > 0.90:
            code = "D002"
        elif defect_prob > 0.87:
            code = "D003"
        elif defect_prob > 0.85:
            code = "D004"
        else:
            code = "D005"
        return {"code": code, "description": DEFECT_CODES[code]}


# ── Convenience function ──────────────────────────────────────────────────────

def run_inspection(image_path: str,
                   model_path: str,
                   config: Dict) -> Dict:
    inspector = QualityInspector(model_path, config)
    return inspector.inspect(image_path)
