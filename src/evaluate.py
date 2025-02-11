import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from diffusers import DDPMScheduler
import os
import torch.nn.functional as F


def evaluate(validation_loader, 
             vae, projection, model,
             high_res_height, high_res_width,
             low_res_height, low_res_width,
             timestep = 0,
             batch_size = 2,):
    
    model.eval()
    vae.eval()
    loss_fn = nn.MSELoss()

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,   # Total number of timesteps
        beta_start=0.0001,          # Starting noise level
        beta_end=0.02,              # Ending noise level
        beta_schedule="linear"      # Noise schedule type
    )
    loss = 0
    n = 0
    with torch.no_grad():
        for low_res, high_res in validation_loader:

            
            timestep = timestep + 1
            low_res, high_res = low_res.to("mps"), high_res.to("mps")
            low_res_resized = torch.nn.functional.interpolate(low_res, size=(high_res_height, high_res_width), mode='bilinear')
            z_high = vae.encode(low_res_resized).latent_dist.sample()
            noise = torch.randn_like(z_high)
            timesteps = torch.randint(timestep, timestep+1, (batch_size,), device="mps").long()
            noisy_z = scheduler.add_noise(z_high, noise, timesteps)
            encoded_low_res = vae.encode(low_res_resized).latent_dist.sample()
            batch_size, channels, h, w = encoded_low_res.shape
            encoded_low_res = encoded_low_res.permute(0, 3, 2, 1).reshape(batch_size, channels, h * w)
            encoded_low_res = projection(encoded_low_res).to("mps")
            predicted_noise = model(noisy_z, timesteps, encoder_hidden_states=encoded_low_res).sample
            denoised_z_batch = []
            for i in range(noisy_z.shape[0]):  
                t = timesteps[i].item()  
                denoised_output = scheduler.step(
                    predicted_noise[i].unsqueeze(0),  
                    t,  
                    noisy_z[i].unsqueeze(0)  
                )
                denoised_z_batch.append(denoised_output.prev_sample)
            denoised_z = torch.cat(denoised_z_batch, dim=0)
            high_res_predicted = vae.decode(denoised_z).sample.clamp(0, 1)
            noisey_low_res = vae.decode(noisy_z).sample.clamp(0, 1)
            high_res_resized = F.interpolate(high_res, size=(536, 600), mode='bilinear', align_corners=False)

            loss += loss_fn(high_res_predicted, high_res_resized).item()
            n = n + 1
            image_dir = f"timescale/{timestep}"
            # os.makedirs(f"{image_dir}", exist_ok=True)
            # save_image(noisey_low_res[0],image_dir + f"/{timestep}_noisey_low_res")
            # save_image(high_res[0],image_dir + f"/{timestep}_high_res")
            # save_image(high_res_predicted[0],image_dir + f"/{timestep}_high_res_predicted")
            # save_image(low_res[0],image_dir + "/low_res")
            # print(f"{n} | {loss/n:.4f}")
           

    return loss/n
def save_image(image, name):
    image = image.cpu().detach()
    image_array = image.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(f'{name}.png')

    