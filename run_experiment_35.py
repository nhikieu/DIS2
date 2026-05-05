'''
DLKD ver 4
- potsdam 512
- some changes to classwise decoder
- changes in KD loss: mse for intermediate and penultimate features, KL for logits
'''

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch.nn.functional as F
from models_bank.PotsdamDataset import PotsdamDataset_noNVDI
# from models.PotsdamDataset import PotsdamDataset
from models_bank.VaihingenDataset import VaihingenDataset
from models_bank import criterion_rs

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import yaml
import time
import wandb
from tqdm import tqdm
from contextlib import contextmanager
import platform
from models_bank.train_util import *
from models_bank.DLKD.DLKD_ver4 import DLKD_ver4
from models_bank.train_util import save_model
from torch.amp import autocast, GradScaler

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

DEVICE = torch.device("cuda:0")

@contextmanager
def memory_management():
    try:
        yield
    finally:
        # Only clear cache when memory usage is high
        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
            torch.cuda.empty_cache()
            
            
def cal_train_loss(dict_results, labels, epoch, device):
    rgb_pred = dict_results['rgb_pred']
    ndsm_pred = dict_results['ndsm_pred']
    kl_loss = dict_results['kl_loss']
    diversity_loss = dict_results['diversity_loss']
    full_seg_pred = dict_results['full_pred']
    miss_seg_pred = dict_results['missing_pred']
    full_scale_preds = dict_results['full_scale_preds']
    miss_scale_preds = dict_results['missing_scale_preds']
    
    full_cross_loss = criterion_rs.softmax_weighted_loss(full_seg_pred, labels)
    full_dice_loss = criterion_rs.dice_loss(full_seg_pred, labels)
    full_loss = full_cross_loss + full_dice_loss
    
    rgb_cross_loss = criterion_rs.softmax_weighted_loss(rgb_pred, labels)
    rgb_dice_loss = criterion_rs.dice_loss(rgb_pred, labels)
    rgb_loss = rgb_cross_loss + rgb_dice_loss
    
    ndsm_cross_loss = criterion_rs.softmax_weighted_loss(ndsm_pred, labels)
    ndsm_dice_loss = criterion_rs.dice_loss(ndsm_pred, labels)
    ndsm_loss = ndsm_cross_loss + ndsm_dice_loss
    
    missing_cross_loss = criterion_rs.softmax_weighted_loss(miss_seg_pred, labels)
    missing_dice_loss = criterion_rs.dice_loss(miss_seg_pred, labels)
    missing_loss = missing_cross_loss + missing_dice_loss

    scale_cross_loss = torch.zeros(1).to(device).float()
    scale_dice_loss = torch.zeros(1).to(device).float()
    for scale in full_scale_preds:
        scale_cross_loss += criterion_rs.softmax_weighted_loss(scale, labels)
        scale_dice_loss += criterion_rs.dice_loss(scale, labels)
    full_scale_loss = scale_cross_loss + scale_dice_loss

    scale_cross_loss = torch.zeros(1).to(device).float()
    scale_dice_loss = torch.zeros(1).to(device).float()
    for scale in miss_scale_preds:
        scale_cross_loss += criterion_rs.softmax_weighted_loss(scale, labels)
        scale_dice_loss += criterion_rs.dice_loss(scale, labels)
    miss_scale_loss = scale_cross_loss + scale_dice_loss

    kl_loss_weight = 5.0 if epoch > 5 else 0.0
    total_loss = full_loss + rgb_loss + ndsm_loss + missing_loss + (kl_loss_weight*kl_loss) + diversity_loss + full_scale_loss + miss_scale_loss

    dict_loss_terms = {
        'total_loss': total_loss,
        'full_loss': full_loss.item(),
        'rgb_loss': rgb_loss.item(),
        'ndsm_loss': ndsm_loss.item(),
        'missing_loss': missing_loss.item(),
        'kl_loss': kl_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'full_scale_loss': full_scale_loss.item(),
        'miss_scale_loss': miss_scale_loss.item(),
    }
    
    return dict_loss_terms


def train_log(dict_train_loss, val_loss, example_ct, log_file_path):
    train_loss = dict_train_loss['total_loss'].item()
    full_loss = dict_train_loss['full_loss']
    rgb_loss = dict_train_loss['rgb_loss']
    ndsm_loss = dict_train_loss['ndsm_loss']
    missing_loss = dict_train_loss['missing_loss']
    kl_loss = dict_train_loss['kl_loss']
    diversity_loss = dict_train_loss['diversity_loss']
    full_scale_loss = dict_train_loss['full_scale_loss']
    miss_scale_loss = dict_train_loss['miss_scale_loss']
    
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'full_loss': full_loss,
        'rgb_loss': rgb_loss,
        'ndsm_loss': ndsm_loss,
        'missing_loss': missing_loss,
        'kl_loss': kl_loss,
        'diversity_loss': diversity_loss,
        'full_scale_loss': full_scale_loss,
        'miss_scale_loss': miss_scale_loss
    }, step=example_ct)
    
    with open(log_file_path, 'a') as f:
        print(f"Train loss after {str(example_ct).zfill(5)} examples: {train_loss:.3f}", file=f)
        print(f"Val loss after {str(example_ct).zfill(5)} examples: {val_loss:.3f}", file=f)
        

def train_batch(images, labels, masks, model, device, epoch):
    model.train()
    
    # Move tensors to device
    images = images.to(device)
    masks = masks.to(device)
    labels = torch.squeeze(labels, dim=1)
    labels = criterion_rs.expand_target(labels)
    labels = labels.type(torch.FloatTensor).to(device)
    
    model.is_training = True
    
    # Use context manager for memory management
    with memory_management():
        dict_results = model(images, masks, device)
        dict_losses = cal_train_loss(dict_results, labels, epoch, device)
    
    # Clear references
    del images, dict_results, labels, masks
    
    return dict_losses


def cal_val_loss(dict_results, labels):
    seg_pred = dict_results['seg_pred']
    if seg_pred.dim() != 4:
        seg_pred = seg_pred.unsqueeze(0)  # Ensure it has batch dimension
    cross_loss = criterion_rs.softmax_weighted_loss(seg_pred, labels)
    dice_loss = criterion_rs.dice_loss(seg_pred, labels)
    total_loss = cross_loss + dice_loss
    
    return total_loss


def validate(model, val_dataloader, device):
    running_val_loss = 0.0
    model.eval()
    model.is_training = False
    
    with torch.no_grad():
        for val_imgs, val_labels, val_masks in val_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = torch.squeeze(val_labels, dim=1)
            val_labels = criterion_rs.expand_target(val_labels)
            val_labels = val_labels.type(torch.FloatTensor).to(device)
            val_masks = val_masks.to(device)
            
            # Group samples by unique masks bc our model can handle one scenario at a time
            unique_masks, inverse_indices = torch.unique(val_masks, dim=0, return_inverse=True)
            grouped_losses = []
            for i in range(unique_masks.size(0)):
                group_indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
                group_imgs = val_imgs[group_indices]
                group_labels = val_labels[group_indices]
                group_masks = val_masks[group_indices]
                dict_results = model(group_imgs, group_masks, device)
                loss = cal_val_loss(dict_results, group_labels).item()
                grouped_losses.append(loss)
            if grouped_losses:
                running_val_loss += (sum(grouped_losses) / len(grouped_losses))
            
        del val_imgs, dict_results, val_labels, val_masks
        
    return running_val_loss / len(val_dataloader)
            

def train(model, train_dataloader, val_dataloader, args, device, checkpoint=None):
    model.to(device)
    scaler = GradScaler()
    
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
    
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        
    best_loss = 100
    
    saveModelPath = args.checkpoints_path
    os.makedirs(saveModelPath, exist_ok=True)
    
    wandb.watch(model, log="all", log_freq=100)
    
    parent_dir = os.path.dirname(args.checkpoints_path)
    
    # win
    if platform.system().lower().startswith('win'):
        checkpoints_id = args.checkpoints_path.split('\\')[-1]
    else:
        checkpoints_id = args.checkpoints_path.split('/')[-1]

    log_file_path = os.path.join(parent_dir, f'training_log_{checkpoints_id}.txt')
    
    example_ct = 0  # number of examples seen
    batch_ct = 0
    mini_batch_size = 2
    
    for epoch in tqdm(range(args.epochs)):
        # TODO change this if train on Vaihingen dataset
        # randomly select a subset of images for training each epoch - 1200 images out of 3k1 images
        if epoch != 0:
            train_dataset = PotsdamDataset_noNVDI(args.train_data_path, amount=1200)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            del train_dataset
            
        batch_ct = 0
        
        mask_batch = None
        for _, (images, labels, masks) in enumerate(train_dataloader):
            
            # if (batch_ct % (mini_batch_size/2)) == 0 or (_+1) == len(train_dataloader):
            scenarios = [
                [False, True],
                [True, False],
            ]
            probs = [0.6, 0.4]  # Example: 60% for [False, True], 40% for [True, False]
            chosen_mask = random.choices(scenarios, weights=probs, k=1)[0]
            mask_batch = torch.tensor([chosen_mask] * images.size(0))
            
            with open(log_file_path, 'a') as f:
                print(f"Chosen mask is {chosen_mask}", file=f)
                
            with open(log_file_path, 'a') as f:
                print(f"Epoch {epoch}, batch {batch_ct}, example {example_ct}", file=f)
            
            with autocast(device_type='cuda'):
                dict_losses = train_batch(images, labels, mask_batch, model, device, epoch)
            
            example_ct +=  len(images)
            
            scaled_loss = dict_losses['total_loss'] / mini_batch_size
            scaler.scale(scaled_loss).backward()
            
            if ((batch_ct % mini_batch_size) == 0) or (_+1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            del images, labels, masks
                
            if ((batch_ct % 16) == 0) or ((_+1) == len(train_dataloader)):
                with memory_management():
                    val_loss = validate(model, val_dataloader, device)
                    
                    train_log(dict_losses, val_loss, example_ct, log_file_path)
                    
                    best_loss = save_model(best_loss, val_loss, saveModelPath, model, optimizer, lr_scheduler)
                    
            batch_ct += 1
            
    return 0


if __name__ == '__main__':
    with open('/home/nhi/Documents/MissingModality/experiments/DLKD/config_potsdam_a6k.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description='DLKD training script')
    parser.add_argument('--train_data_path', type=str, default=config['data']['train_data_path'])
    parser.add_argument('--val_data_path', type=str, default=config['data']['val_data_path'])
    # parser.add_argument('--test_data_path', type=str, default=config['data']['test_data_path'])
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'])
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'])
    parser.add_argument('--checkpoints_path', type=str, default=config['callbacks']['checkpoints_path'])
    
    args = parser.parse_args()
    
    now = str(int(time.time()))
    checkpoints_pth = os.path.join(args.checkpoints_path, now)
    args.checkpoints_path = checkpoints_pth
    
    print(f"dataset train path: {args.train_data_path}")
    print(f"dataset val path: {args.val_data_path}")

    train_dataset = PotsdamDataset_noNVDI(args.train_data_path, amount=1200)
    val_dataset = PotsdamDataset_noNVDI(args.val_data_path, probs=[0.30, 0.30, 0.40])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    wandb.login()
    
    with wandb.init(project="paper_dlkd_missing", entity="nhikieu"):
        config = wandb.config
        config.checkpoints_path = args.checkpoints_path
        
        my_model = DLKD_ver4(num_classes=5, image_size=512)
        
        # potsdam DLKD ver 4 part 1
        # ckpt = r"/home/nhi/Documents/MissingModality/experiments/DLKD/checkpoints/1755494762/1755540780_loss0.6658"
        
        # potsdam DLKD ver 4 part 2 - paused
        # ckpt = r"/home/nhi/Documents/MissingModality/experiments/DLKD/checkpoints/1755560304/1755561950_loss0.6367"
        
        # potsdam DLKD ver 4 part 2
        ckpt = r"/home/nhi/Documents/MissingModality/experiments/DLKD/checkpoints/1755586752/1755589165_loss0.6188"
        
        # potsdam DLKD ver 4 part 3
        # ckpt = r"/home/nhi/Documents/MissingModality/experiments/DLKD/checkpoints/1755644065/1755685803_loss0.6197"

        train(my_model, train_dataloader, val_dataloader, args, DEVICE, ckpt)
    