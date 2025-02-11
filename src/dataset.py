import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def get_datasets(base_path, test_size=.2):

    base_path = "GEOCOLOR_IMAGES"
    paired_images = []
    folders = {
        300: os.path.join(base_path, "300"),
        600: os.path.join(base_path, "600"),
        1200: os.path.join(base_path, "1200"),
        2400: os.path.join(base_path, "2400"),
    }

    
    all_files = {size: glob(f"{folders[size]}/*.png") for size in folders}

    prefixes = set(os.path.basename(f).split("_")[0] for f in all_files[300])
    for prefix in prefixes:
        matched_files = {size: os.path.join(folders[size], f"{prefix}_GOES16-ABI-ne-GEOCOLOR-{size}x{size}.png") for size in folders}
        if all(os.path.exists(matched_files[size]) for size in folders):
            paired_images.append(matched_files)


    train_data, test_data = train_test_split(paired_images, 
                                        test_size = int(len(paired_images) * test_size), 
                                        random_state=2718)
    train_data, validation_data = train_test_split(train_data, 
                                        test_size = int(len(paired_images) * test_size), 
                                        random_state=2718)
    return train_data, validation_data, test_data


       
class SuperResDataset(Dataset):
    def __init__(self, paired_images, low_res_size, high_res_size, transform=None, mode="train",test_size=.2):
       
        self.paired_images = paired_images
        self.low_res_size  = low_res_size
        self.high_res_size = high_res_size
        self.transform = transform

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        img_paths = self.paired_images[idx]
        
        low_res_path = img_paths[self.low_res_size]
        high_res_path = img_paths[self.high_res_size]
        
        low_res = Image.open(low_res_path).convert("RGB")
        high_res = Image.open(high_res_path).convert("RGB")

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res
