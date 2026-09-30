import numpy as np, sys
from PIL import Image
sys.path.insert(0, '/project/peilab/why/eeg-brainit/scripts')
from eval_atm_pipeline import list_test_images
from pathlib import Path
import statistics
gt = list_test_images(Path('/project/peilab/why/data/images_set'))
dirs = {
 'OLD_1024': 'outputs/lowlevel_decoder/sub-08/vae_head/pred_lowlevel_rgb_512',
 'FULL10_512': 'outputs/sdedit_ll_full10/sub-08/vae_head/pred_lowlevel_rgb_512',
 'CONCAT_1536': 'outputs/vae_head_fix/sub-08/head_concat/pred_lowlevel_rgb_512',
}
res={k:[] for k in dirs}
for i in range(200):
    t=np.asarray(Image.open(gt[i]).convert('L').resize((64,64),Image.Resampling.BILINEAR)).astype(np.float64).ravel()
    for k,dp in dirs.items():
        im=np.asarray(Image.open(Path(dp)/f"{i:03d}.png").convert('L').resize((64,64),Image.Resampling.BILINEAR)).astype(np.float64).ravel()
        res[k].append(np.corrcoef(t,im)[0,1])
for k in dirs:
    print(f'{k:14s} gray64 corr={statistics.mean(res[k]):.4f}', flush=True)
