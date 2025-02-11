# To save checkpoints
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image

from evaluate import save_image

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

from dataset import SuperResDataset, get_datasets

import time

def train(train_loader, vae, projection, model,
          high_res_height, high_res_width,
          low_res_height, low_res_width,
          lr = 1e-5, weight_decay=0,
          save_dir = "src/checkpoints",
          num_train_steps = 1000,
          checkpoint_frequency = 10,
          batch_size = 2,
          num_epoch = 1,

          ):
    
    high_res_size = high_res_width
    low_res_size  = low_res_width
    # Loss and noise scheduling
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_steps,  # Total number of timesteps
        beta_start=0.00001,         # Starting noise level
        beta_end=0.02,             # Ending noise level
        beta_schedule="linear"     # Noise schedule type
    )



    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    iterations = 0
    start_time = time.time()
    cum_loss = 0
    cum_timestep = 0
    for epoch in range(num_epoch):
        for low_res, high_res in train_loader:
            low_res, high_res = low_res.to("mps"), high_res.to("mps")

            # Encode high-res image to latent space
            z_high = vae.encode(high_res).latent_dist.sample()

            # Add noise to latent representation
            noise = torch.randn_like(z_high)
            max_noise =  min(max(int((1000*(iterations+1))/(num_train_steps/batch_size)),200),1000)
            timesteps = torch.randint(0, max_noise, (low_res.shape[0],), device="mps")
            noisy_z = scheduler.add_noise(z_high, noise, timesteps.long())
            

            # Resize low_res to match VAE's 600x600
            low_res_resized = torch.nn.functional.interpolate(low_res, size=(high_res_height, high_res_width), mode='bilinear')
            # # Encode low_res into a latent tensor using VAE
            with torch.no_grad():
                encoded_low_res = vae.encode(low_res_resized).latent_dist.sample() 
            
            batch_size, channels, h, w = encoded_low_res.shape
            encoded_low_res = encoded_low_res.permute(0, 3, 2, 1).reshape(batch_size, h * w, channels)
            encoded_low_res = projection(encoded_low_res).to("mps")
            predicted_noise = model(noisy_z, timesteps, encoder_hidden_states=encoded_low_res).sample
            loss = loss_fn(predicted_noise, noise)
            loss.backward()

            
            optimizer.zero_grad()
            optimizer.step()

            print(f"Iteration [{iterations+1}/{int(num_train_steps/batch_size)}] | Loss: {loss.item():.4f} | Timestep {timesteps[0]:.0f}")    


            if iterations % checkpoint_frequency == 0:
               
               
                # cum_loss = cum_loss + loss.item()
                # cum_timestep = cum_timestep + torch.mean(timesteps.float())
                # print(f"Iteration [{iterations+1}/{int(num_train_steps/batch_size)}] | Loss: {cum_loss:.4f} | Timestep {cum_timestep:.0f}")    
                # cum_loss = 0
                # cum_timestep = 0
                
                checkpoint = {
                    "epoch": iterations + 1,
                    "model_state_dict": model.state_dict(),
                    "vae_state_dict": vae.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }
                os.makedirs( f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}", exist_ok=True)
                #torch.save(checkpoint,  f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}/checkpoint_epoch_{iterations+1}.pt")
                torch.save(checkpoint,  f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}/latest4.pt")
                with torch.no_grad():
                    #Save a complete image for visualization
                    denoised_z = noisy_z - predicted_noise
                    high_res_predicted = vae.decode(denoised_z).sample.clamp(0, 1)
                
                    image_dir = f"visual_verification/{low_res_size}_{high_res_size}/{iterations}"
                    os.makedirs(f"{image_dir}", exist_ok=True)
                    high_res_predicted = high_res_predicted[0]
                    high_res = high_res[0]
                    low_res = low_res[0]
                    #save_image(noisy_z[0], f"{image_dir}/noisey_z_{timesteps}")
                    #save_image(predicted_noise[0], f"{image_dir}/predicted_noise{timesteps[0].item()}")
                    save_image(high_res_predicted, f"{image_dir}/predicted_{timesteps[0].item()}")
                    save_image(high_res, f"{image_dir}/high_res")
                    save_image(low_res, f"{image_dir}/low_res")
                    
            
            #del low_res, encoded_low_res, high_res, z_high, noise, noisy_z, predicted_noise
            #del low_res, high_res, z_high, noise, noisy_z, predicted_noise
            #torch.mps.empty_cache()
            
            iterations = iterations + 1
    checkpoint = {
        "epoch": iterations + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    }
    os.makedirs( f"{save_dir}/final/{low_res_size}_{high_res_size}", exist_ok=True)

    torch.save(model.state_dict(),      os.path.join(save_dir+ "/final/", f"{low_res_size}_{high_res_size}/final_model.pt"))
    torch.save(vae.state_dict(),        os.path.join(save_dir + "/final/", f"{low_res_size}_{high_res_size}/final_vae.pt"))
    torch.save(projection.state_dict(), os.path.join(save_dir+"/final/", f"{low_res_size}_{high_res_size}/final_projection.pt"))

    print("Final models saved.")