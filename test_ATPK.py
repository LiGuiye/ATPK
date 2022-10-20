from glob import glob
import numpy as np
import os
from tqdm import tqdm

def sliced_wasserstein(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape  # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(
            A.shape[1], dirs_per_repeat
        )  # (descriptor_component, direction)
        dirs /= np.sqrt(
            np.sum(np.square(dirs), axis=0, keepdims=True)
        )  # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)  # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(
            projA, axis=0
        )  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)  # pointwise wasserstein distances
        results.append(np.mean(dists))  # average over neighborhoods and directions
    return np.mean(results)  # average over repeats

# 4X
scale = 4
data_type = 'Wind'
if data_type == 'Wind':
    original = '/lustre/scratch/guiyli/Dataset_WIND/DIP/Wind2014_removed/u_v'
    gt = '/lustre/scratch/guiyli/Dataset_WIND/DIP/ATP_hr_scale'+str(scale) + '/u_v'
    atpk = '/lustre/scratch/guiyli/Dataset_WIND/DIP/ATP_fake_scale'+str(scale)+ '/u_v'
elif data_type == 'Solar':
    original = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/Solar2014_removed'
    gt = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/ATP_hr_scale'+str(scale)
    atpk = '/lustre/scratch/guiyli/Dataset_NSRDB/DIP/ATP_fake_scale'+str(scale)
image_lsit = glob(original+'/*.npy')
metrics = {'mse': [None] * len(image_lsit), 'std': []}
metrics_min,metrics_max,mse = 0,0,0
name = ''
img_gt,img_atpk = np.empty((2,8*scale,8*scale)),np.empty((2,8*scale,8*scale))

for i, f in enumerate(tqdm(image_lsit)):
    name = os.path.basename(f)[:-4]
    # this might be faster
    img_gt[0] = np.load(gt+'/'+name+'_channel1.npy')
    img_gt[1] = np.load(gt+'/'+name+'_channel2.npy')
    img_atpk[0] = np.load(atpk+'/'+name+'_channel1.npy')
    img_atpk[1] = np.load(atpk+'/'+name+'_channel2.npy')
    # img_gt = np.stack((np.load(gt+'/'+name+'_channel1.npy'), np.load(gt+'/'+name+'_channel2.npy')))
    # img_atpk = np.stack((np.load(atpk+'/'+name+'_channel1.npy'),np.load(atpk+'/'+name+'_channel2.npy')))

    metrics_min = metrics_min if img_gt.min() > metrics_min else img_gt.min()
    metrics_max = metrics_max if img_gt.max() < metrics_max else img_gt.max()

    mse = ((img_gt - img_atpk)**2).mean()
    metrics['mse'][i] = mse / (img_gt.mean()**2)

os.makedirs('results/scale+'+str(scale)+'/'+data_type,exist_ok=True)
np.save(
    os.path.join('results/scale+'+str(scale)+'/'+data_type, "error_mse_"+str(scale)+"X.npy"), metrics['mse']
)

text_file = open(
        os.path.join('results/scale+'+str(scale)+'/'+data_type, "mean_metrics_mse_"+str(scale)+"X.txt"),
        "w",
    )

drange = metrics_max - metrics_min
text_file.write("\n" + "Data Range: " + str(drange) + "\n")
text_file.write(str(metrics_min) + ", " + str(metrics_max) + "\n")

text_file.write("\n" + "MSE/(mean^2) --> mean" + "\n")
text_file.write(str(np.mean(metrics['mse'])) + "\n")

text_file.write("\n" + "MSE/(mean^2) --> median" + "\n")
text_file.write(str(np.median(metrics['mse'])) + "\n")

print("Validation metrics saved!")
