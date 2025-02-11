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
import torch.nn.functional as F


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
        num_train_timesteps=int((num_train_steps/batch_size)*num_epoch),  
        beta_start=0.0001,         
        beta_end=0.02,             
        beta_schedule="linear"     
    )
   
    # Loss functions and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    iterations = 0
    # Learning rate schedule
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=num_epoch*(num_train_steps/batch_size))
    
    # The initial checkpoint
    checkpoint = {
        "epoch": iterations + 1,
        "model_state_dict": model.state_dict(),
        "vae_state_dict": vae.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0,
    }
    

    losses = []
    for epoch in range(num_epoch):
        for low_res, high_res in train_loader:
            low_res, high_res = low_res.to("mps"), high_res.to("mps")

            # Encode high-res image to latent space
            with torch.no_grad():
                z_high = vae.encode(high_res).latent_dist.sample()

                # Add noise to latent representation
                noise = torch.randn_like(z_high)
                max_noise =  min(max(int((100*(iterations+1))/((num_train_steps/batch_size)*num_epoch)),1),1000)
                timesteps = torch.randint(0, max_noise, (batch_size,), device="mps").long()
                noisy_z = scheduler.add_noise(z_high, noise, timesteps)
            

            # Resize low_res to match VAE's 600x600
            low_res_resized = torch.nn.functional.interpolate(low_res, size=(high_res_height,high_res_width), mode='bilinear')
            # Encode low_res into a latent tensor using VAE
            # We won't train the VAE so we use no grad
            with torch.no_grad():
                encoded_low_res = vae.encode(low_res_resized).latent_dist.sample() 
                batch_size, channels, h, w = encoded_low_res.shape
                encoded_low_res = encoded_low_res.permute(0, 3, 2, 1).reshape(batch_size, channels, h * w)
            encoded_low_res = projection(encoded_low_res).to("mps")

            # Model prediction
            predicted_noise = model(noisy_z, timesteps, encoder_hidden_states=encoded_low_res).sample
            # Back prop
            loss = loss_fn(noisy_z, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses.append(loss.item())
            print(f"Iteration [{iterations+1}/{(int(num_train_steps/batch_size)*num_epoch)}] | Loss: {sum(losses[len(losses)-10:])/10:.4f} | Timestep {timesteps[0].item():.0f}")    
            
            if iterations % checkpoint_frequency == 0:
                # Save the model and checkpoint
                torch.save(checkpoint,  f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}/backup.pt")
                checkpoint = {
                    "epoch": iterations + 1,
                    "model_state_dict": model.state_dict(),
                    "vae_state_dict": vae.state_dict(),
                    "projection_state_dict": projection.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }
                os.makedirs( f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}", exist_ok=True)
                torch.save(checkpoint,  f"{save_dir}/checkpoints/{low_res_size}_{high_res_size}/latest.pt")
                # Denoise the images to save
                with torch.no_grad():
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
                    og_image_noisy = vae.decode(noisy_z).sample.clamp(0,1) # This image makes sure the unet is doing something
                    non_noisy = vae.decode(z_high).sample.clamp(0, 1)
                
                    image_dir = f"visual_verification/{low_res_size}_{high_res_size}/{iterations}"
                    os.makedirs(f"{image_dir}", exist_ok=True)
                    save_image(og_image_noisy[0], f"{image_dir}/og_image_noisy")
                    save_image(high_res_predicted[0], f"{image_dir}/predicted")
                    save_image(high_res[0], f"{image_dir}/high_res")
                    save_image(low_res[0], f"{image_dir}/low_res")
                    save_image(non_noisy[0], f"{image_dir}/non_noisy")
                    
            # Manually delete things we don't need
            del low_res, encoded_low_res, high_res, z_high, noise, noisy_z, predicted_noise
            torch.mps.empty_cache()
            iterations = iterations + 1

    # Save the final models
    os.makedirs( f"{save_dir}/final/{low_res_size}_{high_res_size}", exist_ok=True)
    torch.save(model.state_dict(),      os.path.join(save_dir+ "/final/", f"{low_res_size}_{high_res_size}/final_model2.pt"))
    torch.save(vae.state_dict(),        os.path.join(save_dir + "/final/", f"{low_res_size}_{high_res_size}/final_vae2.pt"))
    torch.save(projection.state_dict(), os.path.join(save_dir+"/final/", f"{low_res_size}_{high_res_size}/final_projection2.pt"))
    print("Final models saved.")