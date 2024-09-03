import os
import shutil
import random

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of train, validation, and test ratios must be 1."

    random.seed(seed)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size  

    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(val_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(image_dir, file), os.path.join(test_dir, file))

    print(f"Dataset split complete: {train_size} train, {val_size} validation, {test_size} test images.")

high_res_dir = './images/high_quality_cropped'
low_res_dir = './images/low_quality_0.25x_cropped'
output_high_res_dir = './images/high_quality_cropped_split'
output_low_res_dir = './images/low_quality_0.25x_cropped_split'


split_dataset(high_res_dir, output_high_res_dir)
split_dataset(low_res_dir, output_low_res_dir)

