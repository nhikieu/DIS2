import os
import time
import torch
import numpy as np
import random
import torch
import torch.nn as nn
import kornia.augmentation as K

def save_model(best_loss, val_loss, saveModelPath, model, optimizer, lr_scheduler):
    if val_loss < best_loss:
        best_loss = val_loss
        
        dir = os.listdir(saveModelPath)
        full_path = [os.path.join(saveModelPath, x) for x in dir]
        if len(dir) >= 3:
            # remove oldest checkpoint b4 saving a new one
            oldest_file = min(full_path, key=os.path.getctime)
            os.remove(oldest_file)
        
        f_name = '_'.join([str(int(time.time())), f'loss{best_loss:.4f}'])
        best_model_path = os.path.join(saveModelPath, f_name)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_loss': best_loss
            },
            best_model_path
        )
        
    return best_loss


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    # Retrieve the current base seed (set by torch.manual_seed in main)
    worker_seed = torch.initial_seed() % 2**32
    # Mix in the worker_id to ensure unique randomness per worker
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
    


class GPUAugmenter(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        # Define the pipeline
        # same_on_batch=False ensures every image in the batch gets a DIFFERENT random transform
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0, 
                scale=(0.5, 1.5), 
                p=p, 
                keepdim=True  # Keeps 512x512 (performs padding/cropping automatically)
            ),
            K.RandomRotation(degrees=45.0, p=0.2),
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            
            # CRITICAL: Define what data types you are passing
            # "input" = Image (Bilinear interpolation)
            # "mask"  = Label (Nearest Neighbor interpolation - keeps class integers intact)
            data_keys=["input", "input", "mask"],
            same_on_batch=False
        )

    @torch.no_grad() # Disable gradients for augmentation to save memory
    def forward(self, rgb, dem, label):
        # Kornia expects masks to have a channel dimension: (B, H, W) -> (B, 1, H, W)
        if label.dim() == 3:
            label = label.unsqueeze(1)
            
        # Apply transforms
        rgb_aug, dem_aug, label_aug = self.aug(rgb, dem, label.float())
        
        # Squeeze label back to (B, H, W) and convert to long
        return rgb_aug, dem_aug, label_aug.long().squeeze(1)

