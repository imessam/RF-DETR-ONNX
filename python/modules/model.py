import numpy as np
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Union, NamedTuple
from PIL import Image
import cv2
import onnxruntime as ort
from .onnx_runtime import OnnxRuntimeSession
from .utils import sigmoid, box_cxcywh_to_xywh


@dataclass
class Detection:
    score: float
    label: int
    normalized_box: np.ndarray  # [x, y, w, h] normalized
    unnormalized_box: np.ndarray  # [x, y, w, h] in pixels
    mask: Optional[np.ndarray] = None



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
        self.ort_session_ = OnnxRuntimeSession(model_path, device=device)
        input_shape = self.ort_session_.get_input_shape()
        self.input_height, self.input_width = input_shape[2:]
        
        # Pre-convert normalization constants for speed
        self.means = np.array(self.MEANS, dtype=np.float32).reshape(3, 1, 1)
        self.stds = np.array(self.STDS, dtype=np.float32).reshape(3, 1, 1)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for inference.

        Args:
            image (np.ndarray): Input image (H, W, C) in BGR format.

        Returns:
            np.ndarray: Preprocessed image batch (1, C, H, W).
        """
        # Convert BGR (OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the model's input size
        image = cv2.resize(image, (self.input_width, self.input_height))

        # Convert image to float32 and normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Change dimensions from HWC to CHW before normalization
        image = np.transpose(image, (2, 0, 1))

        # Normalize (vectorized)
        image = (image - self.means) / self.stds

        # Add batch dimension
        image = np.expand_dims(image.astype(np.float32), axis=0)

        return image

    def _post_process(
        self, 
        outputs: list[np.ndarray], 
        origin_height: int, 
        origin_width: int, 
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, 
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES
    ) -> list[Detection]:
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
            list[Detection]: A list of Detection objects.
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
        
        # Filter detections based on the confidence threshold
        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]
        
        # Convert boxes from cxcywh to xywh format (normalized)
        norm_boxes = box_cxcywh_to_xywh(boxes)
        
        # Calculate unnormalized boxes
        unnorm_boxes = norm_boxes.copy()
        unnorm_boxes[..., [0, 2]] *= origin_width
        unnorm_boxes[..., [1, 3]] *= origin_height
        
        # Resize the masks to the original image size if available
        processed_masks = []
        if masks is not None:
            for i in range(len(masks)):
                m = cv2.resize(masks[i], (origin_width, origin_height))
                m = (m > 0).astype(np.uint8) * 255
                processed_masks.append(m)
        
        # Create list of Detection objects
        detections = []
        for i in range(len(scores)):
            mask = processed_masks[i] if processed_masks else None
            detections.append(Detection(
                score=float(scores[i]),
                label=int(labels[i]),
                normalized_box=norm_boxes[i],
                unnormalized_box=unnorm_boxes[i],
                mask=mask
            ))
            
        return detections

    def predict(
        self, 
        image: Union[np.ndarray, Image.Image], 
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD, 
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES
    ) -> tuple[list[Detection], dict[str, float]]:
        """
        Predict bounding boxes and masks for a single image.
        
        Args:
            image: Input image (OpenCV format BGR or PIL Image).
            confidence_threshold: Confidence threshold for filtering boxes.
            max_number_boxes: Maximum number of boxes to return.
            
        Returns:
            A tuple of (detections, timings).
        """
        start_total = time.perf_counter()
        
        # 0. Convert PIL image to OpenCV context if necessary
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        origin_height, origin_width = image.shape[:2]
        
        # 1. Pre-process
        start_pre = time.perf_counter()
        input_tensor = self._preprocess(image)
        end_pre = time.perf_counter()
        
        # 2. Inference
        start_run = time.perf_counter()
        outputs = self.ort_session_.run(input_tensor)
        end_run = time.perf_counter()
        
        # 3. Post-process
        start_post = time.perf_counter()
        detections = self._post_process(
            outputs, 
            origin_height, 
            origin_width, 
            confidence_threshold, 
            max_number_boxes
        )
        end_post = time.perf_counter()
        
        end_total = time.perf_counter()
        
        timings = {
            "preprocess": (end_pre - start_pre) * 1000,
            "ort_run": (end_run - start_run) * 1000,
            "postprocess": (end_post - start_post) * 1000,
            "total": (end_total - start_total) * 1000
        }
        
        return detections, timings

    def save_detections(
        self, 
        image: np.ndarray, 
        detections: list[Detection], 
        save_image_path: str
    ) -> None:
        """
        Draw bounding boxes, masks and class labels on the original image and save it.

        Args:
            image (np.ndarray): Original image (BGR).
            detections (list[Detection]): List of Detection objects.
            save_image_path (str): Path to save the result.
        """
        result = image.copy()
        overlay = image.copy()

        # Generate a color for each unique label (BGR)
        unique_labels = {det.label for det in detections}
        label_colors = {
            label: (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255))
            for label in unique_labels
        }

        # Draw masks on the overlay
        for det in detections:
            if det.mask is not None:
                color = label_colors[det.label]
                mask_bool = det.mask > 0
                overlay[mask_bool] = color

        # Blend the overlay with the original image
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw boxes and labels on the result
        for det in detections:
            label = det.label
            color = label_colors[label]
            box = det.unnormalized_box
            
            # box is [x, y, w, h] float or int, convert to int for cv2
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 4)

            # Draw label text background
            text = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = x + 5
            text_y = y + 5 + text_height
            
            cv2.rectangle(result, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), color, -1)
            
            # Draw label text
            text_color = (255, 255, 255)
            cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Save
        cv2.imwrite(save_image_path, result)
