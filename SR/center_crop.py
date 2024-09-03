from PIL import Image
import os

def center_crop(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2

    return image.crop((left, top, right, bottom))

def process_images(high_res_dir, low_res_dir, output_high_res_dir, output_low_res_dir, low_res_target_size=(64, 64), high_res_target_size=(256, 256)):
    os.makedirs(output_high_res_dir, exist_ok=True)
    os.makedirs(output_low_res_dir, exist_ok=True)

    for low_res_filename in os.listdir(low_res_dir):
        low_res_path = os.path.join(low_res_dir, low_res_filename)
        high_res_path = os.path.join(high_res_dir, low_res_filename)

        if not os.path.exists(high_res_path):
            print(f"High resolution image for {low_res_filename} does not exist.")
            continue

        with Image.open(low_res_path) as low_res_image:
            low_res_cropped = center_crop(low_res_image, low_res_target_size[0], low_res_target_size[1])
            low_res_cropped.save(os.path.join(output_low_res_dir, low_res_filename))

        with Image.open(high_res_path) as high_res_image:
            high_res_cropped = center_crop(high_res_image, high_res_target_size[0], high_res_target_size[1])
            high_res_cropped.save(os.path.join(output_high_res_dir, low_res_filename))

high_res_dir = './images/high_quality'
low_res_dir = './images/low_quality_0.25x'
output_high_res_dir = './images/high_quality_cropped'
output_low_res_dir = './images/low_quality_0.25x_cropped'

process_images(high_res_dir, low_res_dir, output_high_res_dir, output_low_res_dir)
