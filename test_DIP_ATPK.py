import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from osgeo import gdal

def sliced_wasserstein_cuda(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape                                   # (neighborhood, descriptor_component)
    device = torch.device("cuda")
    results = torch.empty(dir_repeats, device=torch.device("cpu"))
    A = torch.from_numpy(A).to(device) if not isinstance(A, torch.Tensor) else A.to(device)
    B = torch.from_numpy(B).to(device) if not isinstance(B, torch.Tensor) else B.to(device)
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.shape[1], dirs_per_repeat, device=device, dtype=torch.float64)          # (descriptor_component, direction)
        dirs = torch.divide(dirs, torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True)))  # normalize descriptor components for each direction
        projA = torch.matmul(A, dirs)                                           # (neighborhood, direction)
        projB = torch.matmul(B, dirs)
        projA = torch.sort(projA, dim=0)[0]                                     # sort neighborhood projections for each direction
        projB = torch.sort(projB, dim=0)[0]
        dists = torch.abs(projA - projB)                                        # pointwise wasserstein distances
        results[repeat] = torch.mean(dists)                                     # average over neighborhoods and directions
    return torch.mean(results)                                                  # average over repeats

def normalize_standard(image):
    """
    Standard Score Normalization

    (image - mean) / std

    return: data_new, mean, std
    """
    if isinstance(image, torch.Tensor):
        mean = torch.mean(image)
        std = torch.std(image)
        return (
            torch.divide(
                torch.add(image, -mean), torch.maximum(std, torch.tensor(1e-5))
            ),
            mean,
            std,
        )
    else:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / max(std, 1e-5), mean, std


# scale_list = [4, 8]
# model_type_list = ['DIP', 'ATPK']
# data_type_list = ['Wind', 'Solar']
scale_list = [4]
model_type_list = ['DIP']
# dip_savePath = 'torch_resize/new_iters'
dip_savePath = 'extract32_iter6k'
data_type_list = ['Wind']
channel = 2
for model_type in model_type_list:
    for data_type in data_type_list:
        for scale in scale_list:
            if model_type == 'ATPK':
                if data_type == 'Wind':
                    original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
                    gt = '/lustre/scratch/guiyli/Dataset_WIND/DIP/ATP_hr_scale'+str(scale) + '/u_v'
                    fake = '/lustre/scratch/guiyli/Dataset_WIND/DIP/ATP_fake_scale'+str(scale)+ '/u_v'
                elif data_type == 'Solar':
                    original = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
                    gt = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/ATP_hr_scale'+str(scale)
                    fake = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/ATP_fake_scale'+str(scale)
            elif model_type == 'DIP':
                if data_type == 'Wind':
                    original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
                    gt = '/lustre/scratch/guiyli/Dataset_WIND/DIP/'+dip_savePath+'/resized_gt_'+str(scale)+'X/u_v'
                    fake = '/lustre/scratch/guiyli/Dataset_WIND/DIP/'+dip_savePath+'/result_dip_'+str(scale)+'X/u_v'
                elif data_type == 'Solar':
                    original = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed/'
                    gt = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/'+dip_savePath+'/resized_gt_'+str(scale)+'X/'
                    fake = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/'+dip_savePath+'/result_dip_'+str(scale)+'X/'

            image_lsit = glob(original+'/*.npy')
            device = torch.device("cuda")
            metrics_mse = torch.empty(len(image_lsit), dtype=torch.float64)
            metrics_swd = torch.empty((channel,len(image_lsit)), dtype=torch.float64)
            metrics_min,metrics_max = 0,0

            for i, f in enumerate(tqdm(image_lsit)):
                name = os.path.basename(f)[:-4]
                if model_type == 'ATPK':
                    img_gt = torch.from_numpy(np.stack((
                            np.load(gt+'/'+name+'_channel1.npy'),
                            np.load(gt+'/'+name+'_channel2.npy')
                            ))).to(device)
                    img_fake = torch.from_numpy(np.stack((
                            np.load(fake+'/'+name+'_channel1.npy'),
                            np.load(fake+'/'+name+'_channel2.npy')
                            ))).to(device)
                elif model_type == 'DIP':
                    img_gt = torch.from_numpy(np.stack((
                        gdal.Open(gt+'/'+name+'_channel0.tif').ReadAsArray(),
                        gdal.Open(gt+'/'+name+'_channel1.tif').ReadAsArray()
                        ))).double().to(device)
                    img_fake = torch.from_numpy(np.stack((
                        gdal.Open(fake+'/'+name+'_channel0.tif').ReadAsArray(),
                        gdal.Open(fake+'/'+name+'_channel1.tif').ReadAsArray()
                        ))).double().to(device)

                metrics_min = metrics_min if img_gt.min() > metrics_min else img_gt.min()
                metrics_max = metrics_max if img_gt.max() < metrics_max else img_gt.max()

                metrics_mse[i] = ((img_gt - img_fake)**2).mean() / (img_gt.mean()**2)

                # SWD
                for c in range(channel):
                    # normalize before calc SWD
                    img_gt_n = normalize_standard(img_gt[c])[0]
                    img_fake_n = normalize_standard(img_fake[c])[0]
                    metrics_swd[c][i] = sliced_wasserstein_cuda(img_gt_n,img_fake_n)

            save_folder = 'Results/test/scale_'+str(scale)+'/'+data_type+'/'+model_type
            if model_type == 'DIP':
                save_folder += '/' + dip_savePath
            os.makedirs(save_folder, exist_ok=True)
            np.save(
                os.path.join(save_folder, "error_mse_"+str(scale)+"X.npy"),
                metrics_mse.numpy()
            )
            np.save(
                os.path.join(save_folder, "error_swd_"+str(scale)+"X.npy"),
                torch.mean(metrics_swd, 0).numpy()
            )

            text_file = open(
                    os.path.join(save_folder, "mean_metrics_mse_swd_"+str(scale)+"X.txt"),
                    "w",
                )

            drange = metrics_max - metrics_min
            text_file.write("\n" + "Data Range: " + str(drange) + "\n")
            text_file.write(str(metrics_min) + ", " + str(metrics_max) + "\n")

            text_file.write("\n" + "MSE/(mean^2) --> mean" + "\n")
            text_file.write(str(torch.mean(metrics_mse).numpy()) + "\n")
            text_file.write("\n" + "MSE/(mean^2) --> median" + "\n")
            text_file.write(str(torch.median(metrics_mse).numpy()) + "\n")

            text_file.write("\n" + "SWD/(mean^2) --> mean" + "\n")
            text_file.write(str(torch.mean(metrics_swd).numpy()) + "\n")
            text_file.write("\n" + "SWD/(mean^2) --> median" + "\n")
            text_file.write(str(torch.median(metrics_swd).numpy()) + "\n")
            print("Validation metrics saved!")
            text_file.close()
