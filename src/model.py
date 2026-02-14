# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2026 PierreMarieCurie
# ------------------------------------------------------------------------

import numpy as np
import random
from typing import Optional
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .onnx_runtime import OnnxRuntimeSession
from .utils import open_image, sigmoid, box_cxcywh_to_xyxyn

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_NUMBER_BOXES = 300

class RFDETRModel:
    """High-level class for RF-DETR model inference."""
    
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, model_path: str, device: str = "gpu"):
        """
        Initialize the RF-DETR model.

        Args:
            model_path (str): Path to the ONNX model file.
            device (str): Device preference ("gpu" or "cpu").
        """
        self.ort_session = OnnxRuntimeSession(model_path, device=device)
        input_shape = self.ort_session.get_input_shape()
        self.input_height, self.input_width = input_shape[2:]

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the input image for inference.

        Args:
            image (Image.Image): Input image.

        Returns:
            np.ndarray: Preprocessed image batch (1, C, H, W).
        """
        # Resize the image to the model's input size
        image = image.resize((self.input_width, self.input_height))

        # Convert image to numpy array and normalize pixel values
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)

        # Change dimensions from HWC to CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def _post_process(
        self, 
        outputs: list[np.ndarray], 
        origin_height: int, 
        origin_width: int, 
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, 
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Post-process the model's output to extract bounding boxes and class information.
        Inspired by the PostProcess class in rfdetr/lwdetr.py: https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/models/lwdetr.py#L701

        Args:
            outputs (list[np.ndarray]): Raw model outputs.
            origin_height (int): Original image height.
            origin_width (int): Original image width.
            confidence_threshold (float): Confidence threshold for filtering.
            max_number_boxes (int): Maximum number of boxes to return.

        Returns:
            tuple: (scores, labels, boxes, masks)
        """
        # Get masks if instance segmentation
        if len(outputs) == 3:  
            masks = outputs[2]
        else:
            masks = None
        
        # Apply sigmoid activation
        prob = sigmoid(outputs[1]) 
        
        # Get detections with highest confidence and limit to max_number_boxes
        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]
        
        # Convert boxes from cxcywh to xyxyn format and scale to image size
        boxes = box_cxcywh_to_xyxyn(boxes)
        boxes[..., [0, 2]] *= origin_width
        boxes[..., [1, 3]] *= origin_height
        
        # Resize the masks to the original image size if available
        if masks is not None:
            new_w, new_h = origin_width, origin_height
            masks = np.stack([
                np.array(Image.fromarray(img).resize((new_w, new_h)))
                for img in masks
            ], axis=0)
            masks = (masks > 0).astype(np.uint8) * 255 
        
        # Filter detections based on the confidence threshold
        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]
        
        return scores, labels, boxes, masks

    def predict(
        self, 
        image_path: str, 
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, 
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Run the model inference and return the detections.

        Args:
            image_path (str): Path or URL to the input image.
            confidence_threshold (float): Confidence threshold.
            max_number_boxes (int): Maximum boxes to return.

        Returns:
            tuple: (scores, labels, boxes, masks)
        """
        # Load the image
        image = open_image(image_path).convert('RGB')
        origin_width, origin_height = image.size
        
        # Preprocess the image
        input_image = self._preprocess(image)

        # Run the model
        outputs = self.ort_session.run(input_image)
        
        # Post-process
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def save_detections(
        self, 
        image_path: str, 
        boxes: np.ndarray, 
        labels: np.ndarray, 
        masks: Optional[np.ndarray], 
        save_image_path: str
    ) -> None:
        """
        Draw bounding boxes, masks and class labels on the original image and save it.

        Args:
            image_path (str): Path to original image.
            boxes (np.ndarray): Bounding boxes (xyxy).
            labels (np.ndarray): Class labels.
            masks (Optional[np.ndarray]): Segmentation masks.
            save_image_path (str): Path to save the result.
        """
        # Load base image
        base = open_image(image_path).convert("RGBA")
        result = base.copy()

        # Generate a color for each unique label (RGBA)
        label_colors = {
            label: (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    100)
            for label in np.unique(labels)
        }

        # Loop over all masks
        if masks is not None:
            for i in range(masks.shape[0]):
                label = labels[i]
                color = label_colors[label]

                # --- Draw mask ---
                mask_overlay = Image.fromarray(masks[i]).convert("L")
                mask_overlay = ImageOps.autocontrast(mask_overlay)
                overlay_color = Image.new("RGBA", base.size, color)
                overlay_masked = Image.new("RGBA", base.size)
                overlay_masked.paste(overlay_color, (0, 0), mask_overlay)
                result = Image.alpha_composite(result, overlay_masked)

        # Convert to RGB for drawing boxes and text
        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        font = ImageFont.load_default()

        # Loop over boxes and draw
        for i, box in enumerate(boxes.astype(int)):
            label = labels[i]
            # Use same color as mask but fully opaque for the outline
            box_color = tuple(label_colors[label][:3])
            draw.rectangle(box.tolist(), outline=box_color, width=4)

            # Draw label text
            text_x = box[0] + 5
            text_y = box[1] + 5
            draw.text((text_x, text_y), str(label), fill=box_color, font=font)

        # Save
        result_rgb.save(save_image_path)
