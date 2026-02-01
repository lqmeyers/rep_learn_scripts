import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import json

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def show_masks_with_colormap(image, masks, scores, box_coords=None, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    cmap = plt.get_cmap('inferno')
    norm = plt.Normalize(vmin=np.min(scores), vmax=np.max(scores))
    for mask, score in sorted(zip(masks, scores), key=lambda x: -x[1]):
        color = cmap(norm(score))
        color = np.array(list(color[:3]) + [0.5])  # RGB + alpha
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        plt.gca().imshow(mask_image)

    if box_coords is not None:
        show_box(box_coords, plt.gca())
    plt.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Score', fontsize=14)
    plt.show()

################################ COCO Format IO ############################################3

def save_coco_json(image_path, nms_boxes, nms_labels, nms_scores, output_path):
    # Get image info
    image = Image.open(image_path)
    width, height = image.size

    # Build COCO structure
    coco_dict = {
        "images": [
            {
                "id": 1,
                "file_name": image_path,
                "width": width,
                "height": height
            }
        ],
        "annotations": [],
        "categories": []
    }

    # Build categories (unique labels)
    label_to_id = {label: idx+1 for idx, label in enumerate(sorted(set(nms_labels)))}
    for label, cat_id in label_to_id.items():
        coco_dict["categories"].append({
            "id": cat_id,
            "name": label
        })

    # Add annotations
    for i, (box, label, score) in enumerate(zip(nms_boxes.cpu().numpy(), nms_labels, nms_scores.cpu().numpy())):
        x_min, y_min, x_max, y_max = box
        width_box = x_max - x_min
        height_box = y_max - y_min
        coco_dict["annotations"].append({
            "id": i+1,
            "image_id": 1,
            "category_id": label_to_id[label],
            "bbox": [float(x_min), float(y_min), float(width_box), float(height_box)],
            "score": float(score),
            "area": float(width_box * height_box),
            "iscrowd": 0
        })

    # Save to file
    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)

def load_coco_nms_boxes_labels_scores(coco_json_path):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Get annotations and categories
    annotations = coco_data["annotations"]
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Extract boxes, labels, and scores
    boxes = []
    labels = []
    scores = []
    for ann in annotations:
        x_min, y_min, width, height = ann["bbox"]
        x_max = x_min + width
        y_max = y_min + height
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(categories[ann["category_id"]])
        scores.append(ann["score"])

    # Convert to tensors
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    return boxes_tensor, labels, scores_tensor
