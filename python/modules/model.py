import numpy as np
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Union, NamedTuple
from PIL import Image, ImageDraw, ImageFont, ImageOps
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
        self.ort_session = OnnxRuntimeSession(model_path, device=device)
        input_shape = self.ort_session.get_input_shape()
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
        outputs = self.ort_session.run(input_tensor)
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
        # Convert BGR to RGBA for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base = Image.fromarray(image_rgb)
        result_rgba = base.convert("RGBA")

        # Generate a color for each unique label (RGBA)
        unique_labels = {det.label for det in detections}
        label_colors = {
            label: (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    100) # Create a semi-transparent overlay for masks
            for label in unique_labels
        }

        overlay_image = Image.new("RGBA", base.size, (0, 0, 0, 0))

        for det in detections:
            label = det.label
            color = label_colors[label]
            
            # Draw mask if available
            if det.mask is not None:
                mask_pil = Image.fromarray(det.mask).convert("L")
                mask_color = Image.new("RGBA", base.size, color)
                overlay_image.paste(mask_color, (0, 0), mask_pil)

        # Composite mask overlay
        result = Image.alpha_composite(result_rgba, overlay_image)
        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        
        font = ImageFont.load_default()

        # Loop over detections and draw boxes
        for det in detections:
            label = det.label
            box = det.unnormalized_box
            
            # Use same color as mask but fully opaque for the outline
            box_color = tuple(label_colors[label][:3])
            
            # box is [x, y, w, h]
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=4)

            # Draw label text
            text_x = x + 5
            text_y = y + 5
            draw.text((text_x, text_y), str(label), fill=box_color, font=font)


        # Save
        result_rgb.save(save_image_path)
