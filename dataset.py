import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import config

class MicroMacroDataset(Dataset):
    def __init__(self, excel_path, image_dir):
        df = pd.read_excel(excel_path)
        df = df[df['Estimated Emotion'] != 'others'].reset_index(drop=True)
        
        self.samples = []
        self.image_dir = image_dir
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for _, row in df.iterrows():
            filename = str(row['Filename']).strip()
            emotion = str(row['Estimated Emotion']).strip()
            if emotion in config.CLASS_TO_IDX:
                label = config.CLASS_TO_IDX[emotion]
                self.samples.append((filename, emotion, label))

    def __len__(self):
        return len(self.samples)

    def compute_optical_flow(self, prev_gray, curr_gray):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Normalize flow channels to [-1, 1]
        flow = np.clip(flow / 15.0, -1.0, 1.0)
        return torch.from_numpy(flow).permute(2, 0, 1).float()  # [2, H, W]

    def __getitem__(self, idx):
        filename, emotion, label = self.samples[idx]
        sample_path = os.path.join(self.image_dir, emotion, filename)
        
        img_files = sorted(
            [f for f in os.listdir(sample_path) if f.lower().endswith(('.jpg', '.png'))],
            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x
        )
        
        indices = np.linspace(0, len(img_files) - 1, config.SEQUENCE_LENGTH).astype(int)
        
        frames = []
        flows = []
        prev_gray = None
        
        for i in indices:
            img_p = os.path.join(sample_path, img_files[i])
            img_pil = Image.open(img_p).convert("RGB")
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (224, 224))
            
            frames.append(self.transform(img_pil))
            
            if prev_gray is None:
                flows.append(torch.zeros(2, 224, 224))
            else:
                flows.append(self.compute_optical_flow(prev_gray, gray))
                
            prev_gray = gray
            
        frames_tensor = torch.stack(frames)  # [16, 3, 224, 224]
        flows_tensor = torch.stack(flows)    # [16, 2, 224, 224]
        
        return frames_tensor, flows_tensor, label