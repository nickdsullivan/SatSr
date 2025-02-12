import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from diffusers import UNet2DConditionModel, AutoencoderKL
from dataset import SuperResDataset, get_datasets

import argparse
from train import train
from evaluate import evaluate


parser = argparse.ArgumentParser(description="Script to run model training")
parser.add_argument('--checkpoint_vae', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--checkpoint_projection', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--checkpoint_model', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--low_res_size', type=int, default=300, help='Size of the low resolution file')
parser.add_argument('--high_res_size', type=int, default=600, help='Size of the high resolution file')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file')
parser.add_argument('--batchsize', type=int, default=1, help='Batchsize')
parser.add_argument('--train', type=bool, default=False, help='train mode')
parser.add_argument('--eval', type=bool, default=False, help='Eval mode')



args = parser.parse_args()
if args.checkpoint != None and args.checkpoint_vae == None:
    args.checkpoint_vae = args.checkpoint
    args.checkpoint_projection = args.checkpoint
    args.checkpoint_model = args.checkpoint

    
# Data setup
# Split and get the images
train_data, validation_data, test_data = get_datasets(base_path="../GEOCOLOR_IMAGES")

transform = transforms.Compose([
    transforms.ToTensor()
])
# Set the size of the images 
train_dataset = SuperResDataset(train_data, low_res_size=args.low_res_size, high_res_size=args.high_res_size, transform=transform)
validation_dataset = SuperResDataset(validation_data, low_res_size=args.low_res_size, high_res_size=args.high_res_size, transform=transform)
test_dataset = SuperResDataset(test_data, low_res_size=args.low_res_size, high_res_size=args.high_res_size, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
validation_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)


# Model definitions
# The vae's latent space is input size / 8.  This will cause a bit of trouble later
# We will need to stretch our 300 x 300 during inference and during embedding.  
# We do this naively through torch.nn.functional.interpolate

low_res_height = args.low_res_size * .9
low_res_width = args.low_res_size
high_res_height = args.high_res_size * .9
high_res_width = args.high_res_size 

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

if args.checkpoint_vae != None:
    checkpoint = torch.load(args.checkpoint_vae)
    vae.load_state_dict(checkpoint["vae_state_dict"])

# This is the projection for the embedded states
projection = nn.Linear(int(high_res_height//8 * high_res_width//8), 1280).to("mps")

if args.checkpoint_projection != None:
    checkpoint = torch.load(args.checkpoint_projection)
    projection.load_state_dict(checkpoint["projection_state_dict"])


model = UNet2DConditionModel(
    sample_size=high_res_width,  # Latent space size.
    in_channels=4,  
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512), 
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"), 
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
)

if args.checkpoint_model != None:
    checkpoint = torch.load(args.checkpoint_model)
    model.load_state_dict(checkpoint["model_state_dict"])







# Send to my mac device  This will have be changed
vae.to("mps")
projection.to("mps")
model.to("mps")



model.train()
projection.train()
print("Models and data loaded. Starting Training.")
if args.train:
    train(train_loader,
        vae = vae, projection = projection, model = model,
        num_train_steps=len(train_dataset), 
        lr = 1e-4, weight_decay=1e-7,
        checkpoint_frequency = 50,
        high_res_height = int(high_res_height),
        high_res_width = int(high_res_width),
        low_res_height = int(low_res_height),
        low_res_width = int(low_res_width),
        batch_size=args.batchsize,
        num_epoch=1)
if args.eval:
    print(evaluate(validation_loader,
            vae = vae, projection = projection, model = model,
            high_res_height = int(high_res_height),
            high_res_width = int(high_res_width),
            low_res_height = int(low_res_height),
            low_res_width = int(low_res_width),))




