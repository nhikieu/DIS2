import torch.nn.functional as F
import torch
import numpy as np


__all__ = ['softmax_weighted_loss', 'dice_loss', 'expand_target']

cross_entropy = F.cross_entropy

def expand_target(x, n_class=5):
    """
    Optimized version using F.one_hot.
    Converts NxHxW label image to NxCxHxW.
    """
    # x is (N, H, W)
    # F.one_hot creates (N, H, W, C)
    # permute moves C to the second dimension -> (N, C, H, W)
    return F.one_hot(x.long(), num_classes=n_class).permute(0, 3, 1, 2).float()

def dice_loss(output, target, eps=1e-7):
    """
    Vectorized Dice Loss. 
    Calculates Dice for all classes simultaneously (in parallel on GPU).
    """
    target = target.float()
    
    # Calculate intersection and union for ALL classes at once
    # Sum over Batch (0), Height (2), and Width (3) dimensions
    # Result shape: (num_cls,)
    intersection = torch.sum(output * target, dim=(0, 2, 3))
    cardinality = torch.sum(output, dim=(0, 2, 3)) + torch.sum(target, dim=(0, 2, 3))
    
    dice_score = 2.0 * intersection / (cardinality + eps)
    
    # Return 1 - mean(dice)
    return 1.0 - torch.mean(dice_score)

def softmax_weighted_loss(output, target):
    """
    Vectorized Softmax Weighted Loss.
    Calculates per-image, per-class weights without loops.
    """
    target = target.float()
    B, C, H, W = output.shape
    
    # 1. Calculate Spatial Sums (Pixels per class per image)
    # Shape: (B, C)
    class_counts = torch.sum(target, dim=(2, 3))
    
    # 2. Calculate Total Pixels per image
    # Shape: (B, 1) - keep dim for broadcasting
    total_pixels = torch.sum(class_counts, dim=1, keepdim=True)
    
    # 3. Vectorized Weight Calculation
    # Formula: 1.0 - (class_count / total_pixels)
    # Shape: (B, C)
    weights = 1.0 - (class_counts / (total_pixels + 1e-7))
    
    # 4. Reshape weights for broadcasting
    # Reshape from (B, C) to (B, C, 1, 1) so it broadcasts over H and W
    weights = weights.view(B, C, 1, 1)
    
    # 5. Calculate Loss
    # Clamp and Log
    log_preds = torch.log(torch.clamp(output, min=0.005, max=1.0))
    
    # Weighted Cross Entropy (Element-wise multiplication)
    # weights (B,C,1,1) * target (B,C,H,W) * log_preds (B,C,H,W)
    cross_loss = -1.0 * weights * target * log_preds
    
    return torch.mean(cross_loss)
  

def rgb_to_class(mask_image):
  class_map = {
    (255,255,255): 0, # #FFFFFF
    (0,0,255): 1, # #0000FF
    (0,255,255): 2, # #00FFFF
    (0,255,0): 3, # #00FF00
    (255,255,0): 4, # #FFFF00
    (255,0,0): 5 # #FF0000
  }

  # Create a 3D numpy array that represents the RGB color of each pixel
  rgb_data = mask_image.reshape(-1, 3)

  # Create a 1D numpy array that represents the class label for each RGB color
  class_labels = np.zeros(rgb_data.shape[0], dtype=np.uint8)
  for rgb, class_label in class_map.items():
      mask = np.all(rgb_data == np.array(rgb), axis=1)
      class_labels[mask] = class_label

  # Reshape the 1D class label array into a 2D class map
  class_data = class_labels.reshape(mask_image.shape[:2])

  return class_data


def class_to_rgb(image):
  valGT = np.array([[255,255,255], [0,0,255], [0,255,255], [0,255,0], [255,255,0], [255,255,255]])

  output = valGT[image]
      
  return output.astype('uint8')


def convert_prediction(image):
  valGT = [[255,255,255], [0,0,255], [0,255,255], [0,255,0], [255,255,0], [255,0,0]]

  output = np.zeros((image.shape[0], image.shape[1], 3))
  
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      class_idx = image[i, j]
      output[i,j,:] = valGT[class_idx]
      
  return output.astype('uint8')
