from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class DeepFashion(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data_names = [os.path.join(root, name) for name in sorted(os.listdir(root))]
        self.len = len(self.data_names)
        self.label_item = []
        for label_img in sorted(os.listdir(root)):
            label_cloth=''
            count=0
            last=0
            for x in label_img:
                if x == '-' and last == 0:
                    last=count
                elif x == '-' and last > 0:
                    label_cloth=label_img[0:count+1]
                    break
                else:
                    count=count+1
            self.label_item.append(label_cloth)

    def __len__(self):
        return self.len 
    
    def __iter__(self):
        return self

    def __getitem__(self, idx):
        img = (Image.open(self.data_names[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(img)
        return (img.clone(),self.label_item[idx])

def dataloader(batch_size, dir="../DeepFashion/train_images"):

    transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    dataset = DeepFashion(root=dir, transform=transform)
    gen = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=40)
    return gen

if __name__ == "__main__":

    dataload = dataloader(4)
    for i, (data,label_clothing) in enumerate(dataload, 0):
        print(label_clothing[0])
