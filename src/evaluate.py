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
             timesteps = 0):
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
            timesteps = timesteps + 20
            low_res, high_res = low_res.to("mps"), high_res.to("mps")
            low_res_resized = torch.nn.functional.interpolate(low_res, size=(high_res_height, high_res_width), mode='bilinear')

            z_high = vae.encode(low_res_resized).latent_dist.sample()
            noise = torch.randn_like(z_high)
            timesteps = torch.tensor(timesteps, dtype=torch.int64).to("mps")

            # Weird stuff that is needed 
            noisy_z = scheduler.add_noise(z_high, noise, timesteps)
            encoded_low_res = vae.encode(low_res_resized).latent_dist.sample()  # Shape: [2, 4, 75, 75]
            batch_size, channels, h, w = encoded_low_res.shape
            encoded_low_res = encoded_low_res.permute(0, 3, 2, 1).reshape(batch_size, channels, h * w)
            encoded_low_res = projection(encoded_low_res).to("mps")
            predicted_noise = model(noisy_z, timesteps, encoder_hidden_states=encoded_low_res).sample

            denoised_z = noisy_z - predicted_noise
            high_res_predicted = vae.decode(denoised_z).sample.clamp(0, 1)

            noisey_low_res = vae.decode(noisy_z).sample.clamp(0, 1)


            # This interpolation is needed for a very interesting reason. 
            # Because we got rid of the text/watermark in the image by remove the botton 10% of the image
            # We were left with a size of 540 x 600.
            # Our vae moves our image into a latent space 1/8 the size of the original image
            # 540 / 8 = 67.5 which is rounded to 67
            # This is 
            high_res_resized = F.interpolate(high_res, size=(536, 600), mode='bilinear', align_corners=False)

            loss += loss_fn(high_res_predicted, high_res_resized).item()
            n = n + 1
            image_dir = f"timescale/{timesteps}"
            os.makedirs(f"{image_dir}", exist_ok=True)
            save_image(noisey_low_res[0],image_dir + f"/{timesteps}_noisey_low_res")
            save_image(high_res[0],image_dir + f"/{timesteps}_high_res")
            save_image(high_res_predicted[0],image_dir + f"/{timesteps}_high_res_predicted")
            save_image(low_res[0],image_dir + "/low_res")
           

    return loss/n
def save_image(image, name):
    image = image.cpu().detach()
    image_array = image.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(f'{name}.png')
    # print("Saved image")
    