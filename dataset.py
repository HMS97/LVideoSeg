import os
import torch
import json
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import one_hot
from path import Path
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, Frames = 10):
        self.root_dirs = Path(root_dir).dirs()
        self.videos = [i for i in self.root_dirs]
        self.Frames = Frames
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        data = {}       
        with open(os.path.join(self.videos[idx],'frame_data.json'), 'r') as f: # open the json file
            video_data = json.load(f)
        video_start_points = video_data['video_start_points']
        frames =  [i for i in  sorted(self.videos[idx].files('*.jpg'), key=lambda x:  int(x.stem.split('_')[-1]))]
        labels = [1 if i in video_start_points else 0 for i in range(len(frames))]
        sliced_list = [(i-(self.Frames*2 + np.random.randint(1,self.Frames) ),i+(self.Frames*2 + np.random.randint(1,self.Frames) )) for i in video_start_points]
        location_label_list = []
        location_frame_list = []
        for start, end in sliced_list:
            location_label_list.extend(labels[start:end+1])
            location_frame_list.extend(frames[start:end+1])
        
        location_frame_list = [read_image(i) for i in location_frame_list]
        location_frame_list = [torch.nn.functional.interpolate(i.unsqueeze(0), size=224).squeeze(0) for i in location_frame_list]
       
        
        data['frames'] = torch.stack(location_frame_list)
        data['labels'] = torch.tensor(location_label_list)
        data['video_start_points'] = video_start_points
        data['path'] = self.videos[idx]
        return data
   

# class VideoDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dirs = Path(root_dir).dirs()
#         self.videos = [i for i in self.root_dirs]
        
#     def __len__(self):
#         return len(self.videos)
    
#     def __getitem__(self, idx):
#         data = {}       
#         frames =  [i for i in  sorted(self.videos[idx].files('*.jpg'), key=lambda x:  int(x.stem.split('_')[-1]))]
#         frames = [read_image(i) for i in frames]
#         frames = [torch.nn.functional.interpolate(i.unsqueeze(0), size=224).squeeze(0) for i in frames]
#         with open(os.path.join(self.videos[idx],'frame_data.json'), 'r') as f: # open the json file
#             video_data = json.load(f)
#         video_start_points = video_data['video_start_points']
#         labels = [1 if i in video_start_points else 0 for i in range(len(frames))]
        
#         data['frames'] = torch.stack(frames)
#         data['labels'] = torch.tensor(labels)
#         data['video_start_points'] = video_start_points
#         data['path'] = self.videos[idx]
#         return data
   
   
# class VideoDataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dirs = Path(root_dir).dirs()
#         self.videos = [i for i in self.root_dirs]
        
#     def __len__(self):
#         return len(self.videos)
    
#     def __getitem__(self, idx):
#         data = {}       
#         frames =  [read_image(i) for i in  sorted(self.videos[idx].files('*.jpg'), key=lambda x:  int(x.stem.split('_')[-1]))]
#         frames = [torch.nn.functional.interpolate(i.unsqueeze(0), size=224).squeeze(0) for i in frames]
#         with open(os.path.join(self.videos[idx],'frame_data.json'), 'r') as f: # open the json file
#             video_data = json.load(f)
#         video_start_points = video_data['video_start_points']
#         labels = []
#         toggle = 0
#         for i in range(len(frames)):
#             if i in video_start_points:
#                 toggle = 1 - toggle # switch between 0 and 1
#             labels.append(toggle)        
            
#         data['frames'] = torch.stack(frames)
#         data['labels'] = torch.tensor(labels)
#         data['video_start_points'] = video_start_points
#         data['path'] = self.videos[idx]
#         return data
   
class Test_VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dirs = Path(root_dir).dirs()
        self.videos = [i for i in self.root_dirs]
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        data = {}        
        frames =  [read_image(i) for i in  sorted(self.videos[idx].files())]
        # print(sorted(self.videos[idx].files()))
        frames = [torch.nn.functional.interpolate(i.unsqueeze(0), size=224).squeeze(0) for i in frames]
        # print(frames)
        # with open(os.path.join(self.videos[idx],'frame_data.json'), 'r') as f: # open the json file
        #     video_data = json.load(f)
        # video_start_points = video_data['video_start_points']
        # labels = [1 if i in video_start_points else 0 for i in range(len(frames))]
        
        data['frames'] = torch.stack(frames)
        # data['labels'] = torch.tensor(labels)
        # data['video_start_points'] = video_start_points
        data['path'] = self.videos[idx]
        return data
   