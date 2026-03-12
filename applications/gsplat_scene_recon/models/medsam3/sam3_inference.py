# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SAM3 inference wrapper for medical image segmentation.

Wraps the upstream ``sam3`` package (pip-installed from
https://github.com/facebookresearch/sam3) with convenience methods for
text-prompted and box-prompted segmentation.

The upstream ``sam3`` package must be installed separately — either via
``pip install git+https://github.com/facebookresearch/sam3.git`` or by
following the instructions in the project README.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sam3 as _sam3_pkg
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor


def normalize_bbox(bbox_xywh, img_w, img_h):
    """Normalize bounding box coordinates to [0, 1] range.

    Inlined from ``sam3.visualization_utils`` to avoid pulling in heavy
    transitive dependencies (pandas, pycocotools, scikit-image) that the
    upstream module imports at the top level.
    """
    if isinstance(bbox_xywh, list):
        normalized_bbox = bbox_xywh.copy()
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
    else:
        assert isinstance(bbox_xywh, torch.Tensor), "Only torch tensors are supported for batching."
        normalized_bbox = bbox_xywh.clone()
        assert normalized_bbox.size(-1) == 4, "bbox_xywh tensor must have last dimension of size 4."
        normalized_bbox[..., 0] /= img_w
        normalized_bbox[..., 1] /= img_h
        normalized_bbox[..., 2] /= img_w
        normalized_bbox[..., 3] /= img_h
    return normalized_bbox


_SAM3_PACKAGE_ROOT = Path(_sam3_pkg.__file__).resolve().parent


class SAM3Model:
    """Wrapper for SAM3 model inference."""

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize SAM3 model.

        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cuda' or 'cpu')
            checkpoint_path: Path to custom checkpoint file (optional).
                            If None, loads default SAM3 from HuggingFace.
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    def load_model(self):
        """Load SAM3 model (lazy loading)."""
        if self.model is not None:
            return

        if self.checkpoint_path:
            print(f"Loading SAM3 model from checkpoint: {self.checkpoint_path}")
        else:
            print("Loading SAM3 model from HuggingFace...")

        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        bpe_path = _SAM3_PACKAGE_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if self.checkpoint_path:
            # For custom checkpoints (e.g., MedSAM3), we need to handle
            # different checkpoint formats. First build the model without
            # loading weights, then load the custom checkpoint.
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=None,  # Don't load checkpoint here
                load_from_HF=False,  # Don't load from HF
            )
            # Load custom checkpoint with flexible format handling
            self._load_custom_checkpoint(self.checkpoint_path)
        else:
            # Use default HuggingFace loading
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path), checkpoint_path=None, load_from_HF=True
            )

        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)

        print("SAM3 model loaded successfully!")

    def _load_custom_checkpoint(self, checkpoint_path: str):
        """
        Load custom checkpoint with flexible format handling.

        Handles both:
        - SAM3 format: keys with 'detector.' prefix
        - MedSAM3 format: keys without 'detector.' prefix
        """
        print(f"Loading custom checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Extract model state dict
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Check if keys have 'detector.' prefix
        sample_key = list(state_dict.keys())[0] if state_dict else ""
        has_detector_prefix = "detector." in sample_key

        if has_detector_prefix:
            # SAM3 format: strip 'detector.' prefix
            clean_state_dict = {
                k.replace("detector.", ""): v for k, v in state_dict.items() if "detector" in k
            }
        else:
            # MedSAM3 format: keys already match model structure
            clean_state_dict = state_dict

        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(clean_state_dict, strict=False)

        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    def encode_image(self, image: np.ndarray) -> dict:
        """
        Encode an image for inference.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Inference state dictionary
        """
        self.load_model()

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Encode image (autocast scoped to inference only; context properly exited)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(pil_image)

        return inference_state

    def predict_box(
        self, inference_state: dict, bbox: Tuple[int, int, int, int], img_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Run inference with bounding box prompt.

        Args:
            inference_state: Image encoding state
            bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
            img_size: Image size as (height, width)

        Returns:
            Binary prediction mask or None if no prediction
        """
        self.processor.reset_all_prompts(inference_state)

        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = img_size

        # Convert to xywh format
        width = x_max - x_min
        height = y_max - y_min

        # Convert xywh to cxcywh
        box_xywh = torch.tensor([x_min, y_min, width, height], dtype=torch.float32).view(1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)

        # Normalize to [0, 1]
        norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()

        # Run inference (autocast scoped so context is properly exited)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            box_state = self.processor.add_geometric_prompt(
                state=inference_state, box=norm_box, label=True  # Positive prompt
            )

        if box_state["masks"] is not None and len(box_state["masks"]) > 0:
            best_idx = torch.argmax(box_state["scores"]).item()
            pred_mask = box_state["masks"][best_idx].cpu().numpy() > 0
            return pred_mask.astype(np.uint8)

        return None

    def predict_text(self, inference_state: dict, text_prompt: str) -> Optional[np.ndarray]:
        """
        Run inference with text prompt.

        Args:
            inference_state: Image encoding state
            text_prompt: Natural language description

        Returns:
            Binary prediction mask or None if no prediction
        """
        self.processor.reset_all_prompts(inference_state)

        # Run text-prompted inference (autocast scoped so context is properly exited)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_state = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        if text_state["masks"] is not None and len(text_state["masks"]) > 0:
            best_idx = torch.argmax(text_state["scores"]).item()
            pred_mask = text_state["masks"][best_idx].cpu().numpy() > 0
            return pred_mask.astype(np.uint8)

        return None

    def get_confidence(self, state: dict) -> float:
        """Get confidence score from state."""
        if state["scores"] is not None and len(state["scores"]) > 0:
            best_idx = torch.argmax(state["scores"]).item()
            return state["scores"][best_idx].item()
        return 0.0

    def predict_text_union(
        self,
        inference_state: dict,
        text_prompt: str,
        score_threshold: float = 0.3,
        max_masks: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Run inference with text prompt and return union of masks above threshold.

        Args:
            inference_state: Image encoding state
            text_prompt: Natural language description
            score_threshold: Minimum score for mask inclusion
            max_masks: Maximum number of masks to union (0 = all)

        Returns:
            Binary prediction mask (union of qualifying masks) or None
        """
        self.processor.reset_all_prompts(inference_state)

        # Run text-prompted inference (autocast scoped so context is properly exited)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_state = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        if text_state["masks"] is None or len(text_state["masks"]) == 0:
            return None

        masks = text_state["masks"]
        scores = text_state["scores"]

        # Filter by score threshold
        if scores is not None:
            scores_np = scores.float().cpu().numpy()
            keep = scores_np >= score_threshold
            if keep.any():
                masks = [m for m, k in zip(masks, keep) if k]
                scores_np = scores_np[keep]
            else:
                # Fallback: keep best scoring mask
                best_idx = int(np.argmax(scores_np))
                masks = [masks[best_idx]]
                scores_np = scores_np[best_idx : best_idx + 1]

            # Limit number of masks
            if max_masks > 0 and len(masks) > max_masks:
                top_idx = np.argsort(scores_np)[-max_masks:]
                masks = [masks[i] for i in top_idx]

        if len(masks) == 0:
            return None

        # Union all masks
        combined = None
        for m in masks:
            m_np = (m.float().cpu().numpy() > 0).astype(np.uint8)
            if combined is None:
                combined = m_np
            else:
                combined = np.maximum(combined, m_np)

        return combined

    def predict_box_union(
        self,
        inference_state: dict,
        bbox: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        score_threshold: float = 0.3,
        max_masks: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Run inference with bounding box prompt and return union of masks above threshold.

        Args:
            inference_state: Image encoding state
            bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
            img_size: Image size as (height, width)
            score_threshold: Minimum score for mask inclusion
            max_masks: Maximum number of masks to union (0 = all)

        Returns:
            Binary prediction mask (union of qualifying masks) or None
        """
        self.processor.reset_all_prompts(inference_state)

        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = img_size

        # Convert to xywh format
        width = x_max - x_min
        height = y_max - y_min

        # Convert xywh to cxcywh
        box_xywh = torch.tensor([x_min, y_min, width, height], dtype=torch.float32).view(1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)

        # Normalize to [0, 1]
        norm_box = normalize_bbox(box_cxcywh, img_w, img_h).flatten().tolist()

        # Run inference (autocast scoped so context is properly exited)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            box_state = self.processor.add_geometric_prompt(
                state=inference_state, box=norm_box, label=True  # Positive prompt
            )

        if box_state["masks"] is None or len(box_state["masks"]) == 0:
            return None

        masks = box_state["masks"]
        scores = box_state["scores"]

        # Filter by score threshold
        if scores is not None:
            scores_np = scores.float().cpu().numpy()
            keep = scores_np >= score_threshold
            if keep.any():
                masks = [m for m, k in zip(masks, keep) if k]
                scores_np = scores_np[keep]
            else:
                # Fallback: keep best scoring mask
                best_idx = int(np.argmax(scores_np))
                masks = [masks[best_idx]]
                scores_np = scores_np[best_idx : best_idx + 1]

            # Limit number of masks
            if max_masks > 0 and len(masks) > max_masks:
                top_idx = np.argsort(scores_np)[-max_masks:]
                masks = [masks[i] for i in top_idx]

        if len(masks) == 0:
            return None

        # Union all masks
        combined = None
        for m in masks:
            m_np = (m.float().cpu().numpy() > 0).astype(np.uint8)
            if combined is None:
                combined = m_np
            else:
                combined = np.maximum(combined, m_np)

        return combined

    def encode_image_batch(self, images: list) -> list:
        """
        Encode multiple images for batch inference.

        Args:
            images: List of RGB images as numpy arrays (H, W, 3)

        Returns:
            List of inference state dictionaries
        """
        self.load_model()
        states = []
        for image in images:
            pil_image = Image.fromarray(image)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = self.processor.set_image(pil_image)
            states.append(inference_state)
        return states


def generate_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Generate bounding box from binary mask.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    return (x_min, y_min, x_max, y_max)


def resize_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a binary mask to target shape.

    Args:
        mask: Binary mask
        target_shape: (height, width)

    Returns:
        Resized mask
    """
    mask = np.squeeze(mask)

    # Convert to PIL Image for resizing
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)

    return (np.array(mask_resized) > 127).astype(np.uint8)


if __name__ == "__main__":
    # Test SAM3 inference
    print("Testing SAM3 inference...")
    print("=" * 60)

    # Import dataset loaders
    from dataset_loaders import load_dataset

    # Initialize model
    sam3 = SAM3Model(confidence_threshold=0.1)

    # Test on one sample from each dataset
    for dataset_name in ["CHASE_DB1", "CVC-ClinicDB"]:
        print(f"\n{dataset_name}:")

        samples = list(load_dataset(dataset_name, max_samples=1))
        if not samples:
            print("  No samples found")
            continue

        sample = samples[0]
        print(f"  Sample ID: {sample.sample_id}")
        print(f"  Image shape: {sample.image.shape}")

        # Encode image
        inference_state = sam3.encode_image(sample.image)

        # Generate bbox from GT
        bbox = generate_bbox_from_mask(sample.gt_mask)
        if bbox is None:
            print("  No bbox generated from GT")
            continue

        print(f"  GT BBox: {bbox}")

        # Box prompt inference
        img_size = sample.gt_mask.shape
        pred_mask_box = sam3.predict_box(inference_state, bbox, img_size)

        if pred_mask_box is not None:
            # Resize if needed
            if pred_mask_box.shape != img_size:
                pred_mask_box = resize_mask(pred_mask_box, img_size)
            print(f"  Box prediction shape: {pred_mask_box.shape}")
            print(f"  Box prediction sum: {pred_mask_box.sum()}")
        else:
            print("  Box prediction: None")

        # Text prompt inference
        pred_mask_text = sam3.predict_text(inference_state, sample.text_prompt)

        if pred_mask_text is not None:
            if pred_mask_text.shape != img_size:
                pred_mask_text = resize_mask(pred_mask_text, img_size)
            print(f"  Text prediction shape: {pred_mask_text.shape}")
            print(f"  Text prediction sum: {pred_mask_text.sum()}")
        else:
            print("  Text prediction: None")

    print("\n" + "=" * 60)
    print("SAM3 inference test complete!")
