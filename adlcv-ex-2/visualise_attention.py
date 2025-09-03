import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from vit import ViT
import torch
from imageclassification import prepare_dataloaders

model = ViT(
    image_size=(32, 32),
    patch_size=(4, 4),
    channels=3,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    pos_enc='learnable',
    pool='cls',
    dropout=0.3,
    fc_dim=None,
    num_classes=2
)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

images, labels = next(iter(prepare_dataloaders(batch_size=8)[0]))  # trainloader

image = images[0:1]
label = labels[0:1]

if torch.cuda.is_available():
    image, label = image.cuda(), label.cuda()

with torch.no_grad():
    logits, attentions = model(image, return_all_attentions=True)

layer_idx = 0 # first layer 
attn = attentions[layer_idx]  


# we take the attention from CLS token to all patches
cls_attn = attn[:, 0, 1:]  # shape (num_heads, seq_len - 1)
cls_attn = cls_attn.mean(dim=0)  # shape (seq_len - 1,)

H, W = 32, 32
patch_h, patch_w = 4, 4
nph, npw = H // patch_h, W // patch_w

cls_attn_2d = cls_attn.view(nph, npw).detach().cpu().numpy()

plt.figure(figsize=(4, 4))
plt.title("Attention for Layer 0 (CLS -> Patches)")
plt.imshow(cls_attn_2d, cmap='viridis')
plt.colorbar()
plt.show()


# Convert attention to 1 x 1 x nph x npw
attn_map = torch.tensor(cls_attn_2d).unsqueeze(0).unsqueeze(0)  # shape: (1,1,nph,npw)
attn_map = F.interpolate(attn_map, scale_factor=patch_h, mode='nearest')  # shape: (1,1,H, W)
attn_map = attn_map.squeeze().cpu().numpy()

# Convert your image back to numpy for plotting
# (assuming your image was normalized in [-1,1] or something similar)
img_np = image[0].permute(1, 2, 0).cpu().numpy()  # shape: (H,W,C)
# Undo your normalization if needed. E.g. if you used mean=0.5, std=0.5
img_np = (img_np * 0.5 + 0.5).clip(0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(img_np)
plt.imshow(attn_map, cmap='rainbow', alpha=0.5)  # overlay
plt.title("Layer 0, CLS Attention Overlay")
plt.axis('off')
plt.show()
