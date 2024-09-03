import cv2
import os
import numpy as np
from calculate_weight import calculate_weight


def load_image(path):
    return cv2.imread(path)


def save_image(image, path):
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])


def weighted_average(images, weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    output_image = np.zeros_like(images[0], dtype=np.float32)

    for img, weight in zip(images, weights):
        output_image += img * weight

    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image


def image_fusion():
    dataset = "test"
    
    image_dirs = ['/home/BasicSR/src/BasicSR/results/EDSR_Lx4_test/visualization/test_dataset/',
             '/home/BasicSR/src/BasicSR/results/ESRGAN_SRx4_test/visualization/test_dataset/',
             '/home/BasicSR/src/BasicSR/results/SwinIR_SRx4_test/visualization/test_dataset/',
             '/home/BasicSR/src/BasicSR/results/RCAN_x4_test/visualization/test_dataset/',
             '/home/SR/src/MambaIR/results/MambaIR_SR_x4_test/visualization/test_dataset/']
    model_names = ["_EDSR_Lx4_test", "_ESRGAN_SRx4_test", "_SwinIR_SRx4_test", '_RCAN_x4_test', "_MambaIR_SR_x4_test"]
    original_image_dir = "/home/SR/src/MambaIR/datasets/images/" + dataset + "/HR/"
    
    for filename in os.listdir(original_image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_paths = []
            for idx in range(len(image_dirs)):
                image_name = filename[:-4] + model_names[idx] + ".png"
                image_path = image_dirs[idx] + image_name
                image_paths.append(image_path)
            
            images = [load_image(path) for path in image_paths]
            print("-" * 10 + "begin to fuse image : " + filename + "-" * 10)
            model_index, model_name, weight_list = calculate_weight(filename, dataset)
            
            output_folder = '/home/fusion_images/' + dataset
            output_image = weighted_average(images, weight_list)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            output_path = '/home/fusion_images/' + dataset + "/" + filename
            save_image(output_image, output_path)
            print("*" * 70)


if __name__ == "__main__":
    image_fusion()
