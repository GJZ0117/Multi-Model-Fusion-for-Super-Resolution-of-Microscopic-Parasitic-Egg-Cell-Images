import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

lpips_model = lpips.LPIPS(net='alex')  

def calculate_metrics(original_image_path, generated_image_path):
    psnr_values = []
    ssim_values = []
    lpips_values = []


    if not os.path.exists(generated_image_path):
        print(f"Generated image for {generated_image_path} does not exist.")

    original_img = Image.open(original_image_path).convert('RGB')
    generated_img = Image.open(generated_image_path).convert('RGB')

    original_np = np.array(original_img)
    generated_np = np.array(generated_img)

    current_psnr = psnr(original_np, generated_np)

    min_dim = min(original_np.shape[0], original_np.shape[1])
    win_size = min(7, min_dim // 2 * 2 + 1)  
    current_ssim = ssim(original_np, generated_np, multichannel=True, win_size=win_size, channel_axis=-1)

    original_tensor = lpips.im2tensor(lpips.load_image(original_image_path))  
    generated_tensor = lpips.im2tensor(lpips.load_image(generated_image_path))  
    current_lpips = lpips_model(original_tensor, generated_tensor)

    
    print("PSNR: ", current_psnr)
    print("SSIM: ", current_ssim)
    print("LPIPS: ", current_lpips.item())

    return current_psnr, current_ssim, current_lpips.item()


if __name__ == "__main__":

    original_image = '/home/SR/src/MambaIR/datasets/images/test/HR/Ascaris lumbricoides_0001.jpg'

    # generated_image = '/home/SR/src/MambaIR/results/MambaIR_SR_x4_test/visualization/test_dataset/Ascaris lumbricoides_0001_MambaIR_SR_x4_test.png'
    # generated_image = '/home/BasicSR/src/BasicSR/results/EDSR_Lx4_test/visualization/test_dataset/Ascaris lumbricoides_0001_EDSR_Lx4_test.png'
    # generated_image = '/home/BasicSR/src/BasicSR/results/ESRGAN_SRx4_test/visualization/test_dataset/Ascaris lumbricoides_0001_ESRGAN_SRx4_test.png'
    # generated_image = '/home/BasicSR/src/BasicSR/results/SwinIR_SRx4_test/visualization/test_dataset/Ascaris lumbricoides_0001_SwinIR_SRx4_test.png'
    # generated_image = '/home/BasicSR/src/BasicSR/results/RCAN_x4_test/visualization/test_dataset/Ascaris lumbricoides_0001_RCAN_x4_test.png'
    generated_image = '/home/fusion_images/test/Ascaris lumbricoides_0001.jpg'
    calculate_metrics(original_image, generated_image)
