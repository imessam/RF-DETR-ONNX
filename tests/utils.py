import numpy as np

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def match_detections(ref_dets, test_dets, iou_threshold=0.5):
    """
    Greedy matching between reference and test detections.
    Returns: list of (ref_det, test_det, iou)
    """
    matches = []
    unmatched_ref = list(range(len(ref_dets)))
    unmatched_test = list(range(len(test_dets)))
    
    if not ref_dets or not test_dets:
        return matches, unmatched_ref, unmatched_test
        
    # Calculate all IoUs
    ious = np.zeros((len(ref_dets), len(test_dets)))
    for i in range(len(ref_dets)):
        for j in range(len(test_dets)):
            if ref_dets[i]['class_id'] == test_dets[j]['class_id']:
                ious[i, j] = calculate_iou(ref_dets[i]['bbox'], test_dets[j]['bbox'])
    
    while unmatched_ref and unmatched_test:
        best_iou = -1
        best_ref = -1
        best_test = -1
        
        for i in unmatched_ref:
            for j in unmatched_test:
                if ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_ref = i
                    best_test = j
        
        if best_iou < iou_threshold:
            break
            
        matches.append((ref_dets[best_ref], test_dets[best_test], best_iou))
        unmatched_ref.remove(best_ref)
        if best_test in unmatched_test:
            unmatched_test.remove(best_test)
        
    return matches, unmatched_ref, unmatched_test
