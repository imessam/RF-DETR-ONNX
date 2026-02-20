import numpy as np

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format."""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    x_inter1 = max(x1_1, x1_2)
    y_inter1 = max(y1_1, y1_2)
    x_inter2 = min(x2_1, x2_2)
    y_inter2 = min(y2_1, y2_2)
    
    intersection = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    area1 = w1 * h1
    area2 = w2 * h2
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
