from PIL import Image
from matplotlib import pyplot as plt

def crop_square(image_path, crop_size, downscale=0, translation=(0,0)):
    """
    Crop a square from the center of the image with optional translation.

    Args:
        image_path (str): Path to the image file.
        crop_size (int): The side length of the square crop.
        downscale (int): Factor to downscale the image before cropping.
        translation (tuple): (x, y) translation to apply to the crop.

    Returns:
        cropped_image: The cropped PIL Image.
    """
    img = Image.open(image_path).convert("RGB")
    if downscale > 0:
        new_size = (img.width // downscale, img.height // downscale)
        img = img.resize(new_size, Image.ANTIALIAS)

    left = (img.width - crop_size) / 2 + translation[0]
    top = (img.height - crop_size) / 2 + translation[1]
    right = (img.width + crop_size) / 2 + translation[0]
    bottom = (img.height + crop_size) / 2 + translation[1]

    cropped_image = img.crop((left, top, right, bottom))
    return cropped_image


def visualize_bounding_boxes(image, boxes, labels=None, scores=None):
    """
    Visualize bounding boxes on an image.

    Args:
        image_path (str): Path to the image file.
        boxes (list of list of float): List of bounding boxes, each box is [x_min, y_min, x_max, y_max].
        labels (list of str, optional): List of labels for each bounding box.
        scores (list of float, optional): List of scores for each bounding box.
    """
    # Load the image
    # image = Image.open(image_path)
    # Get image size in inches for matplotlib (pixels / dpi)
    dpi = 100  # You can adjust this value if needed
    width, height = image.size
    fig, ax = plt.subplots(1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(image)

    # Add bounding boxes
    for i, box in enumerate(boxes.cpu().numpy()):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Add label and score if available
        if labels and scores != None:
            label = f"{labels[i]}: {scores[i]:.2f}"
        elif labels:
            label = labels[i]
        elif scores:
            label = f"{scores[i]:.2f}"
        else:
            label = None

        if label:
            plt.text(x_min, y_min, label, fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.show()