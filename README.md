# SatSr
A diffusion-based method for super-resolution of satellite images. The goal is to enhance small images' resolution and improve image quality.

Data Collection: I used GOES-16, data, specifically the most recent batch of geocolor images of the Northeast U.S. Due to hardware limitations, I focused on upscaling 300×300 images to 600×600. The dataset was obtained from a website that frequently updates available images. The exact URL I scraped the files from is: NOAA GOES-16, accessed on February 7, 2025, at 5:41 PM. 
Relevant code

Data Processing: Each image was converted to PNG using Pillow. During training, I noticed that the images contained a watermark and a bottom section with date and time, which added unnecessary complexity. To address this, I removed the bottom 10% of each image, resulting in a non-square format. This introduces some challenges, which I will discuss later.
Relevant code


Technique: I am using a technique called latent space diffusion, which has two main components. The first part is the variational autoencoder (VAE). I use a pre-trained VAE, which consists of an encoder-decoder pair. As the name suggests, the encoder maps an image into a latent space, while the decoder reconstructs an image from that latent representation.  The other main component is the denoising U-Net, which takes an encoded image in latent space and removes noise from it. I use a conditional denoising U-Net, meaning it also takes an additional input into its hidden state. In Stable Diffusion, this auxiliary input is typically a text prompt for image generation. In my model, I provide a clear low-resolution image as the conditioning input.
I am using a technique called latent space diffusion which has two main components.  The first part is the variational autoencoder (VAE).  I use a pre-trained VAE which consists of an encoder, and a decoder model.  As the name suggests, the encoder encodes the image into a latent space while the decoder decodes the latent space into an image.  The other main part is the denoising U-Net, which takes an encoded image in the latent space and removes noise from it.  I am using a conditional denoising U-Net, meaning it has a second hidden state input.  In Stable Diffusion, this auxiliary input is typically a text prompt for image generation. In my model, I provide a clear low-resolution image as the conditioning input.

Training:  I used a pre-trained VAE but an untrained conditional denoising U-Net. The VAE encoded each image into a latent space 1/8th of its original size with four channels. Because of this, the U-Net needed to be trained for a specific set of input sizes. Since I was running this locally, I limited training to super-resolution from 300×300 to 600×600.

I have left a few example images on my GitHub page. I used a DDPM schedule to generate noise, with Adam as the optimizer and MSE as the loss function. The model was trained overnight for 5 epochs.

I used a pre-trained vae but untrained a denoising conditional unet.  The VAE encoded each image into a latent space 1/8th of its original size, and into 4 channels.  Because of this the U-Net needed to be trained for a specific set of input sizes.  Since I was running this locally, I limited training to super-resolution from 300×300 to 600×600.  I used a DDPM schedule to generate noise, with Adam as the optimizer and MSE as the loss function. The model was trained overnight for 5 epochs.

As mentioned in the data processing section, I removed the bottom 10% of each image. This resulted in training images of size 300×270 and 600×540. Since the VAE encodes images into a latent space 1/8th of the original size, the latent dimensions for 600×540 become approximately 75×67 (since 540/8 = 67.5, which is rounded). When decoding, the output image size is 600×536, so I used Torch’s interpolate function to resize it to 600×540.

Training Process:
Load a batch of images into memory.
Encode the high-resolution images using the VAE's encoder.
Add noise at a random timestep between 0 and the current training iteration.
Encode the low-resolution images for use as the hidden state.
Input the noisy high-resolution image and the non-noisy low-resolution image into the denoising U-Net.
Compute the MSE loss between the predicted noise and the actual noise.
Perform gradient descent on the denoising U-Net, keeping the VAE frozen.
Inference Process:
Load a batch of low-resolution images into memory.
Create a copy of each low-resolution image and stretch it to match the high-resolution size.
Encode the stretched images using the VAE's encoder.
Add low-level noise (at 0.01 intensity).
Encode the original (unstretched) low-resolution images for the hidden state.
Input the noisy latent image and the non-noisy low-resolution image into the denoising U-Net.
Decode the denoised output using the VAE's decoder.
Save the final super-resolved image.

Relevant code


Results: The average MSE on the testset was ~0.14 at a noise level of 0.0095.  I believe with more training I would have better results.   I will train the model for longer tonight and will upload new images in the morning.  Here are some examples of images.  They look pretty good but it is clear they could be better.