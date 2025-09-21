import os, csv, random
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

def rgb_to_yuv(img_tensor):
    R, G, B = img_tensor[0], img_tensor[1], img_tensor[2]
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.14713*R - 0.28886*G + 0.436*B
    V = 0.615*R - 0.51499*G - 0.10001*B
    return torch.stack([Y, U, V], dim=0)

class PilotNetDataset(Dataset):
    def __init__(self, root_dir, csv_path, max_angle_rad=None,
                 crop=(0,0,0,0), resize=(66,200), augment=True):
        # 👇 Use the folder containing labels.csv as the root
        self.root = os.path.dirname(csv_path)

        self.items = []
        self.max_angle = max_angle_rad
        self.crop, self.resize, self.augment = crop, resize, augment

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                img = row["image"]  # e.g. "images/img_000000.png"
                angle = float(row.get("steer_rad", 0.0))
                if self.max_angle:
                    angle /= self.max_angle
                self.items.append((img, angle))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        img_rel, y = self.items[idx]
        # 👇 This now resolves to: dataset/clear_xxx/images/img_000000.png
        path = os.path.join(self.root, img_rel)

        pil = Image.open(path).convert("RGB")

        # Crop sky/hood if wanted
        t,b,l,r = self.crop
        if any((t,b,l,r)):
            W,H = pil.size
            pil = pil.crop((l,t,W-r,H-b))

        # Augmentations
        if self.augment:
            if random.random() < 0.5:
                factor = 0.5 + random.random()
                pil = ImageEnhance.Brightness(pil).enhance(factor)
            if random.random() < 0.5:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                y = -y

        # Resize to (66,200)
        pil = pil.resize((self.resize[1], self.resize[0]), Image.BILINEAR)

        # To tensor in [0,1]
        img = torch.tensor(list(pil.getdata()), dtype=torch.float32).view(
            pil.size[1], pil.size[0], 3) / 255.0
        img = img.permute(2,0,1)  # (3,H,W)

        # Convert to YUV and normalize
        img = rgb_to_yuv(img)
        img = img - img.mean(dim=(1,2), keepdim=True)

        # Clamp normalized targets to [-1,1]
        if self.max_angle:
            y = max(min(y, 1.0), -1.0)

        return img, torch.tensor([y], dtype=torch.float32)
