import os
import torch
import json
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import one_hot
from path import Path

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dirs = Path(root_dir).dirs()
        self.videos = [i for i in self.root_dirs]
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        data = {}        
        frames =  [read_image(i) for i in  self.videos[idx].files('*.jpg')]
        frames = [torch.nn.functional.interpolate(i.unsqueeze(0), size=224).squeeze(0) for i in frames]
        with open(os.path.join(self.videos[idx],'frame_data.json'), 'r') as f: # open the json file
            video_data = json.load(f)
        video_start_points = video_data['video_start_points']
        labels = [1 if i in video_start_points else 0 for i in range(len(frames))]
        
        data['frames'] = torch.stack(frames)
        data['labels'] = torch.tensor(labels)
        data['video_start_points'] = video_start_points
        data['path'] = self.videos[idx]
        return data
   