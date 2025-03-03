import os
import cv2
import json
import logging
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from network.baseline_network import ATBSN, N_BSN
from network.multi_target_T import Inf_T_elements
from utils.dataloader import Trainingset_Loader, Testset_Loader_SIDD, Testset_Loader_DND, Result_Loader
from utils.function import APR, Recharger
from utils.utility import L_APR, L_RD, CustomCosineLR
from utils.metric import CAL_PSNR, CAL_SSIM

with open("./config.json", "r") as f:
    config = json.load(f)
    
def setup_device():
    torch.cuda.set_device(0)  # 기본적으로 첫 번째 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available_gpus = torch.cuda.device_count()
    if config["gpu_use"] > available_gpus:
        logging.info(f"GPU_use ({config['gpu_use']}) exceeds available GPU count ({available_gpus}). Setting GPU_use to ({available_gpus}).")
        config["gpu_use"] = available_gpus
    return device, list(range(config["gpu_use"]))

def initialize_models(device, device_ids, trainingset_name=None):
    f = ATBSN().to(device)
    fD = N_BSN().to(device)
    inf_T = Inf_T_elements(f) if trainingset_name == None else None
    if config["gpu_use"] > 1:
        f = torch.nn.DataParallel(f, device_ids=device_ids)
        fD = torch.nn.DataParallel(fD, device_ids=device_ids)
        inf_T = torch.nn.DataParallel(inf_T, device_ids=device_ids) if trainingset_name == True else None
    if trainingset_name != None:
        f_model = f.module if isinstance(f, torch.nn.DataParallel) else f
        fD_model = fD.module if isinstance(fD, torch.nn.DataParallel) else fD
        f_model.load_state_dict(torch.load(os.path.join(config["pretrained_path"], f'BSN_{trainingset_name}.pth'))["model_state_dict"])   
        fD_model.load_state_dict(torch.load(os.path.join(config["pretrained_path"], f'NBSN_{trainingset_name}.pth'))["model_state_dict"])   
        f.eval()
        fD.eval()
        return [f, 'BSN'], [fD, 'NBSN']
    else:
        return [f, 'BSN'], [fD, inf_T, 'NBSN']

def load_noisy_dataset(set_name, training=True):
    if training == True:
        dataset = Trainingset_Loader(root_dir=config["dataset_path"]["training"][set_name], 
                                 patch_size=config["patch_size"]["BSN"], n_patches=config["n_patches"], augmentation=True)

    if training == False:
        if 'SIDD' in set_name:
            dataset = Testset_Loader_SIDD(root_dir=config["dataset_path"]["inference"][set_name])
        elif 'DND' in set_name:
            dataset = Testset_Loader_DND(root_dir=config["dataset_path"]["inference"][set_name])
    return [dataset, set_name]

def setup_training(f, fD):
    apr = APR()
    recharger = Recharger()
    optim_f = Adam(f.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    optim_fD = Adam(fD.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    sche_f = CustomCosineLR(optim_f, total_iter=config["total_iter"]["BSN"], current_iter=0, eta_min=0)
    sche_fD = CustomCosineLR(optim_fD, total_iter=config["total_iter"]["NBSN"], current_iter=0, eta_min=0)
    return [apr, optim_f, sche_f, L_APR()], [recharger, optim_fD, sche_fD,  L_RD()]

def save_model(model, save_path):
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    torch.save({
        'model_state_dict': model.module.state_dict() if config["gpu_use"] > 1 else model.state_dict()
    }, save_path)
    logging.info(f"save: {save_path}")
    
def train(trainingset_info, model_info, training_info, device):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    batch_size = config["batch_size"]["Training"]
    dataloader = trainingset_info[0].get_dataloader(batch_size=batch_size, shuffle=False)
    batch_count = (len(trainingset_info[0]) // batch_size)
    
    if model_info[-1] == 'BSN':
        logging.info(f"{model_info[-1]}(APR) training of {trainingset_info[1]} start ----------")
        apr, optim_f, sche_f, l_apr = training_info
        f = model_info[0]

        for epoch in range(config["total_iter"][model_info[-1]] // batch_count + 1):
            for iter_epoch, batch in enumerate(dataloader, start=1):
                optim_f.zero_grad()
                y = batch.to(device)
                
                fy = f(y, hole_size=config["shift_factor"]["training"]*2-1)
                h1_y, h2_y, h1_fy, h2_fy = apr.apply(y, fy)
                f_h1_y = f(h1_y, hole_size=config["shift_factor"]["training"]*2-1)
                loss_apr = l_apr(f_h1_y, h2_y, h1_fy, h2_fy, reg_factor=config["reg_factor"])
                loss_apr.backward()
                
                optim_f.step()
                sche_f.step()
                
                iter_ = batch_count * epoch + iter_epoch
                
                if iter_ % config["logging_iter"] == 0:
                    logging.info(f'APR Iteration: {iter_}, Loss: {loss_apr.item():.5f}, LR: {sche_f.get_last_lr()[0]}')
                if iter_ == config["total_iter"]["BSN"]:
                    save_model(f, os.path.join(config["trained_path"], f'{model_info[-1]}_{trainingset_info[1]}.pth'))
                    logging.info(f"{model_info[-1]}(APR) training of {trainingset_info[1]} complete ----------")
                    return
                
    else:
        logging.info(f"{model_info[-1]}(RD) training of {trainingset_info[1]} start ----------")
        fD = model_info[0]
        inf_T = model_info[1]
        recharger, optim_fD, sche_fD, l_rd = training_info
        
        trained_atbsn_para = torch.load(os.path.join(config["trained_path"], f'BSN_{trainingset_info[1]}.pth'))
        if config["gpu_use"] > 1:
            inf_T.module.Trained_ATBSN.load_state_dict(trained_atbsn_para['model_state_dict'])
        else:
            inf_T.Trained_ATBSN.load_state_dict(trained_atbsn_para['model_state_dict'])

        for epoch in range(config["total_iter"][model_info[-1]] // batch_count + 1):
            for iter_epoch, batch in enumerate(dataloader, start=1):
                optim_fD.zero_grad()
                y = batch[:, :, (config["patch_size"]["BSN"]-config["patch_size"]["NBSN"])//2:config["patch_size"]["BSN"] - (config["patch_size"]["BSN"]-config["patch_size"]["NBSN"])//2, 
                          (config["patch_size"]["BSN"]-config["patch_size"]["NBSN"])//2:config["patch_size"]["BSN"] - (config["patch_size"]["BSN"]-config["patch_size"]["NBSN"])//2].to(device)
                
                with torch.no_grad():
                    t_elements = inf_T(y, config["shift_factor"]["distillation"], padding=config["padding"])
                t_rd_elements = recharger.apply(y, t_elements)
                fDy = fD(y)
                loss_rd = l_rd(fDy, t_rd_elements)
                loss_rd.backward()
                
                optim_fD.step()
                sche_fD.step()
                
                iter_ = batch_count * epoch + iter_epoch
                
                if iter_ % config["logging_iter"] == 0:
                    logging.info(f'RD Iteration: {iter_}, Loss: {loss_rd.item():.5f}, LR: {sche_fD.get_last_lr()[0]}')
                if iter_ == config["total_iter"]["NBSN"]:
                    save_model(fD, os.path.join(config["trained_path"], f'{model_info[-1]}_{trainingset_info[1]}.pth'))
                    logging.info(f"{model_info[-1]}(RD) training of {trainingset_info[1]} complete ----------")
                    return   
                
def inference(trainingset_name, testset_info, model_info, device):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    if 'SIDD' in testset_info[1]:
        batch_size = config["batch_size"]["Inference"]["SIDD (BSN)"] if model_info[-1] == 'BSN' else config["batch_size"]["Inference"]["SIDD (NBSN)"]
    else:
        batch_size = config["batch_size"]["Inference"]["DND (BSN)"] if model_info[-1] == 'BSN' else config["batch_size"]["Inference"]["DND (NBSN)"]
    dataloader = testset_info[0].get_dataloader(batch_size=batch_size, shuffle=False)
    
    save_dir = os.path.join(config["result_path"], model_info[-1], f'train_{trainingset_name}', f'test_{testset_info[1]}')
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f'{model_info[-1]} inference of {testset_info[1]} start')
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            y = batch.to(device)

            if model_info[-1] == 'BSN':
                y = F.pad(y, (config["padding"], config["padding"], config["padding"], config["padding"]), mode='reflect')
                fy = model_info[0](y, hole_size=config["shift_factor"]["inference"]*2-1)[:, :, config["padding"]:-config["padding"], config["padding"]:-config["padding"]] # f(y)
            else:
                fy = model_info[0](y) # fD(y)

            fy = fy.cpu().numpy()
            fy = np.transpose(np.clip(fy, 0, 1), [0,2,3,1])

            if 'SIDD' in testset_info[1]:
                fy = np.uint8(np.round(fy * 255))
                for ea in range(batch_size):
                    filename = f"{(idx * batch_size + ea + 1):04d}.png"
                    fy_img = Image.fromarray(fy[ea])
                    fy_img.save(os.path.join(save_dir, filename))

            else:
                for ea in range(batch_size):
                    group_num = (idx * batch_size + ea) // 20 + 1
                    img_num = (idx * batch_size + ea) % 20 + 1
                    filename = f"{group_num:04d}_{img_num:02d}.mat"

                    fy_mat = fy[ea]
                    sio.savemat(os.path.join(save_dir, filename), {'Idenoised_crop': fy_mat})  
    logging.info(f'{model_info[-1]} inference of {testset_info[1]} complete')  
                    
def validate():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    GT_images = Testset_Loader_SIDD(root_dir=config["dataset_path"]["inference"]["SIDD_Validation"], GT=True)
    result_images_BSN = Result_Loader(os.path.join(config["result_path"], "BSN/train_SIDD_Medium/test_SIDD_Validation"))
    result_images_NBSN = Result_Loader(os.path.join(config["result_path"], "NBSN/train_SIDD_Medium/test_SIDD_Validation"))
    
    pnsr_mean_BSN = CAL_PSNR(GT_images, result_images_BSN)
    ssim_mean_BSN = CAL_SSIM(GT_images, result_images_BSN)
    logging.info(f'BSN => PSNR of SIDD_Validation: {pnsr_mean_BSN:.4f}(DB) / SSIM of SIDD_Validation: {ssim_mean_BSN:.4f}')
    
    pnsr_mean_NBSN = CAL_PSNR(GT_images, result_images_NBSN)
    ssim_mean_NBSN = CAL_SSIM(GT_images, result_images_NBSN)
    logging.info(f'NBSN => PSNR of SIDD_Validation: {pnsr_mean_NBSN:.4f}(DB) / SSIM of SIDD_Validation: {ssim_mean_NBSN:.4f}')
    return