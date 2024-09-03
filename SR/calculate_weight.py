from calculate_metrics import calculate_metrics

def calculate_weight(filename, dataset="test"):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    model_name = ["EDSR", "ESRGAN", "SwinIR", "RCAN", "MambaIR"]
    model_name_prefix = ["_EDSR_Lx4_test", "_ESRGAN_SRx4_test", "_SwinIR_SRx4_test", "_RCAN_x4_test","_MambaIR_SR_x4_test"]
    model_index = [i for i in range(len(model_name))]
    result_image_dir  = ["/home/BasicSR/src/BasicSR/results/EDSR_Lx4_test/visualization/test_dataset/",
                        "/home/BasicSR/src/BasicSR/results/ESRGAN_SRx4_test/visualization/test_dataset/",
                        "/home/BasicSR/src/BasicSR/results/SwinIR_SRx4_test/visualization/test_dataset/",
                        "/home/BasicSR/src/BasicSR/results/RCAN_x4_test/visualization/test_dataset/",
                        "/home/SR/src/MambaIR/results/MambaIR_SR_x4_test/visualization/test_dataset/"]
    for idx in range(len(result_image_dir)):
        result_image_dir[idx] = result_image_dir[idx] + filename[:-4] + model_name_prefix[idx] + ".png"

    original_image_dir = ["/home/BasicSR/src/BasicSR/datasets/images/" + dataset + "/HR/",
                         "/home/BasicSR/src/BasicSR/datasets/images/" + dataset + "/HR/",
                         "/home/BasicSR/src/BasicSR/datasets/images/" + dataset + "/HR/",
                         "/home/BasicSR/src/BasicSR/datasets/images/" + dataset + "/HR/",
                         "/home/SR/src/MambaIR/datasets/images/" + dataset + "/HR/"]
    for idx in range(len(result_image_dir)):
        original_image_dir[idx] = original_image_dir[idx] + filename
    

    for idx in range(len(original_image_dir)):
        print("#" * 10 + "calculate " + model_name[idx] + " metrics" + "#" * 10)
        test_image_path = original_image_dir[idx]
        generate_image_path = result_image_dir[idx]
        psnr, ssim, lpips = calculate_metrics(test_image_path, generate_image_path)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        lpips_scores.append(lpips)

    lpips_scores_inverse = [1 / score for score in lpips_scores]

    weight_list = []

    while len(weight_list) == 0:

        for i in range(len(psnr_scores)):
            psnr_weight = round(psnr_scores[i] / sum(psnr_scores), 2) * 10
            ssim_weight = round(ssim_scores[i] / sum(ssim_scores), 2) * 10
            lpips_weight = round(lpips_scores_inverse[i] / sum(lpips_scores_inverse), 2) * 10
            weight_list.append(pow(5, round((psnr_weight + ssim_weight + lpips_weight) / 3, 2)))

        weight_sum = sum(weight_list)
        for i in range(len(weight_list) - 1):
            weight_list[i] = round(weight_list[i] / weight_sum, 2)
        weight_list[len(weight_list) - 1] = round(weight_list[len(weight_list) - 1] / weight_sum, 2)


        min_case_weight = 1
        bad_case_index = -1

        for weight_index in range(len(weight_list)):
            if weight_list[weight_index] < 0.2 and weight_list[weight_index] < min_case_weight:
                bad_case_weight = weight_list[weight_index]
                bad_case_index = weight_index

        if bad_case_index != -1:
            del model_index[bad_case_index]
            del model_name[bad_case_index]
            del psnr_scores[bad_case_index]
            del ssim_scores[bad_case_index]
            del lpips_scores[bad_case_index]
            weight_list = []

    print("final models and weights : ")
    print(model_name)
    print(weight_list)

    return model_index, model_name, weight_list


if __name__ == "__main__":
    calculate_weight("Ascaris lumbricoides_0001.jpg", "test")
