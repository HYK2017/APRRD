import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def CAL_PSNR(GT_list, denoised_list, data_range=255):
    if len(denoised_list) != len(GT_list):
        raise ValueError("The lengths of the two lists are not equal.")
    
    psnr_sum = 0
    for denoised_img, gt_img in zip(denoised_list, GT_list):
        denoised_img = (denoised_img*255).numpy()
        gt_img = (gt_img*255).numpy()
        denoised_img = np.transpose(denoised_img, (1, 2, 0))
        gt_img = np.transpose(gt_img, (1, 2, 0))
        
        psnr_sum = psnr_sum + peak_signal_noise_ratio(denoised_img, gt_img, data_range=data_range)
    return psnr_sum/len(denoised_list)
    
def CAL_SSIM(GT_list, denoised_list, data_range=255, gaussian_weights=True, channel_axis=2, use_sample_covariance=False):
    if len(denoised_list) != len(GT_list):
        raise ValueError("The lengths of the two lists are not equal.")
    
    ssim_sum = 0
    for denoised_img, gt_img in zip(denoised_list, GT_list):
        denoised_img = (denoised_img*255).numpy()
        gt_img = (gt_img*255).numpy()
        denoised_img = np.transpose(denoised_img, (1, 2, 0))
        gt_img = np.transpose(gt_img, (1, 2, 0))
        
        ssim_sum = ssim_sum + structural_similarity(denoised_img, gt_img, 
                                                    data_range=data_range, gaussian_weights=gaussian_weights, channel_axis=channel_axis, use_sample_covariance=use_sample_covariance) 
    return ssim_sum/len(denoised_list)