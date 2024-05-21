import cv2
import tifffile as tiff
import numpy as np
import os
import random
from PIL import Image


def read_tif_with_tifffile(tif_path):
    image = tiff.imread(tif_path)
    if image is None:
        print("Error: Image not found or failed to load.")
        return None
    # Normalize the array to range [0, 255]
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)
    # Convert to HxWxC format (height, width, channels)
    if norm_image.shape[0] == 3:  # Check if first dimension is channels
        norm_image = np.transpose(norm_image, (1, 2, 0))
    return norm_image

def save_image_with_pillow(image, path):
    try:
        im = Image.fromarray(image)
        im.save(path)
        return True
    except Exception as e:
        print(f"Error saving image with Pillow: {e}")
        return False


def random_crop_rotate(image_path, output_folder, label_folder, num_crops):
    # Load the image
    image = read_tif_with_tifffile(image_path)
    if image is None:
        print("Error: Image not found or failed to load.")
        return

    # Ensure the image is 10980x10980
    height, width, channels = image.shape
    if height != 10980 or width != 10980:
        print("Error: The image is not 10980x10980 in size.")
        return

    # Define the size of the crop
    crop_size = 512

    # Prepare output data folders
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    for i in range(num_crops):
        # Randomly select the top-left corner of the crop
        x = random.randint(0, width - crop_size)
        y = random.randint(0, height - crop_size)

        # Crop the image
        cropped_image = image[y:y+crop_size, x:x+crop_size]

        # Check if the cropped image is valid
        if cropped_image is None or cropped_image.size == 0:
            print(f"Error: Cropped image {i+1} is invalid.")
            continue

        # Randomly select a rotation angle
        angle = random.uniform(0, 360)

        # Get the center of the image for rotation
        center = (crop_size // 2, crop_size // 2)

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Ensure image data is in correct format (convert to BGR if necessary)
        if rotated_image.shape[2] == 3:
            rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)

        # Print image details before saving
        print(f"Rotated image {i+1} details: shape={rotated_image.shape}, dtype={rotated_image.dtype}")

        # Save the rotated image using Pillow
        output_image_path = os.path.join(output_folder, f'rotated_image_{i+1}.png')
        print(f"Saving rotated image to: {output_image_path}")
        success = save_image_with_pillow(rotated_image, output_image_path)
        if not success:
            print(f"Error: Failed to save rotated image {i+1}")

        # Save the rotation angle and crop position in a separate label file
        label_file_path = os.path.join(label_folder, f'rotated_image_{i+1}.txt')
        with open(label_file_path, 'w') as f:
            f.write(f"Rotation Angle: {angle}\n")
            f.write(f"Crop Position: (x={x}, y={y})\n")

    print("Processing complete. Files saved to:", output_folder)
    print("Labels saved to:", label_folder)


# Example usage
image_path = 'E:\\遥感影像机器识别定位\\enshi1.tif'
output_folder = 'E:\\遥感影像机器识别定位\\Datasets\\Images'
label_folder = 'E:\\遥感影像机器识别定位\\Datasets\\labels'
num_crops = 50  # Number of random crops to perform

random_crop_rotate(image_path, output_folder, label_folder, num_crops)
