# from re import S
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import dotsi
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import neptune
from einops import rearrange
from lion_pytorch import Lion
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import OrderedDict
from os import path as osp
from path import Path
from model import *
import datetime
from pprint import pprint
import logging
import torch._dynamo
import torch._inductor.config
from dataset import VideoDataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import json
from box import Box
from torch.utils.data import random_split
from torchvision.ops import sigmoid_focal_loss
from lion_pytorch import Lion

torch.backends.cudnn.benchmark = True

torch._dynamo.config.log_level = logging.ERROR

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# kwargs = GradScalerKwargs(init_scale=165536)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class NEP():
    def __init__(self) :
        proxies = {
        'http': os.environ.get('http_proxy'),
        'https': os.environ.get('https_proxy'),
        }

        self.run = neptune.init_run(
            project="csuhuimingsun/NDoising",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ODBjNWQwZS00Mzk1LTRmOGMtYTYwNS05NjY5NjIwNjk0OWUifQ==",
            source_files = ["**/*.py"],
            proxies = proxies
        ) 


def train(args):

    with open('./config/config.json', 'r') as f:
        config_dict = json.load(f)

    config = Box(config_dict)
    
    accelerator = Accelerator( gradient_accumulation_steps =1,  mixed_precision = 'fp16', kwargs_handlers=[ddp_kwargs] )
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = VideoDataset( root_dir='/mnt/drive1/brick1/hsun/videoSeg/data/video_datasets/CondensedMovies/new_videos', Frames = config.model.Frames)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=8)
    train_size = int(0.995 * len(dataset))  # 80% of the dataset
    # train_size = int(0.995 * len(dataset))  # 80% of the dataset

    test_size = len(dataset) - train_size

    # Split the dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])

  


    cfg= dotsi.Dict(vars(args))

  #current date 
    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d')
    save_path = os.path.join(cfg.save_dir,cfg.model_name + f'_{cfg.prefix}_{nowDate}')

    if accelerator.is_local_main_process: 
        pprint(vars(args))
        nep = NEP()
        nep.run['cfg'] = vars(args)
    os.makedirs('logs', exist_ok = True)
    with open(f'logs/{Path(save_path).name}.txt','a') as f:
        f.write(f'seting {args}' +'\n')
    train_loader = DataLoader(train_set, batch_size = 1, num_workers=8 , pin_memory = True,  prefetch_factor= 2)
    val_loader = DataLoader(test_set, batch_size = 1, num_workers=8 , pin_memory = True,)
    pos_weight = torch.tensor([0.5])  # adjust this value as needed

    criterion = nn.BCELoss(weight=pos_weight).cuda()
        
    model = eval(cfg.model_name)(config.model)
    # model = torch.compile(model)

    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume_path), strict=False)
        print(f'resume loaded model from {cfg.resume_path}')



    # optimizer = Lion(model.parameters(), lr=cfg.lr/3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10*len(train_loader), gamma=0.95)

    interval = config.model.Frames
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader,val_loader, scheduler)
    os.makedirs(save_path, exist_ok=True)
    for epoch in  range(cfg.resume, cfg.epochs+1):
    # for epoch in  range(cfg.resume, cfg.resume+1):

        current_video = None

        model.train()
    
        for index,data in enumerate(tqdm(train_loader, disable= not accelerator.is_local_main_process)):
            try:
                model.clear_cache()
            except:
                model.module.clear_cache()
            total_loss = 0
            with accelerator.accumulate(model):
                frames = data['frames'].float().cuda()
                labels = data['labels'].float().cuda()
                video_start_points = data['video_start_points'] 
                path = data['path'] 
                for i in range(0, frames.shape[1], interval):
                    # print(i+interval , frames.shape)
                    if i + interval <= frames.shape[1]:
      
                        current_frames_slice = frames[:, i:i+interval, :, :, :]
                        # gt = labels[:, i:i+interval ]
                        if i == 0 or interval:
                            gt = labels[:, i:i+interval ]
                        else:
                            gt = labels[:, i-interval:i ]

                        pred = model(current_frames_slice)
                        if i > 2*interval:
                            loss = criterion(pred, gt) + 1e-6
                            # if accelerator.is_local_main_process:     
                            #     print(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), loss.detach().cpu().numpy())
                        # total_loss += loss
                            accelerator.backward(loss)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            scheduler.step()




        if accelerator.is_local_main_process:     
            if epoch % cfg.save_interval == 0 :
                torch.save(model.module.state_dict(), os.path.join(save_path, f'{cfg.prefix}_{cfg.model_name}_{epoch}.pth'))
                nep.run['save_path'].log(os.path.join(save_path, f'{cfg.prefix}_{cfg.model_name}_{epoch}.pth'))

            torch.save(model.module.state_dict(), os.path.join(save_path, f'last.pth'))
            print('epoch {} done'.format(epoch), os.path.join(save_path, f'last.pth'))

        # if accelerator.is_local_main_process:     
        #     if epoch % cfg.test_interval  == 0 or epoch ==1 :
        #         model.eval()
        #         correct = 0
        #         total = 0
        #         with torch.no_grad():
        #             # Loop over validation data
        #             for index, data in enumerate(tqdm(val_loader, disable= not accelerator.is_local_main_process)):
        #                 try:
        #                     model.clear_cache()
        #                 except:
        #                     model.module.clear_cache()

        #                 frames = data['frames'].float()
        #                 labels = data['labels'].float()
        #                 video_start_points = data['video_start_points'] 
        #                 path = data['path'] 

        #                 for i in range(0, frames.shape[1], interval):
        #                     if i + interval <= frames.shape[1]:
        #                         frames_slice = frames[:, i:i+interval, :, :, :]
        #                         gt = labels[:, i:i+interval]

        #                         # Forward pass
        #                         outputs = model(frames_slice)
        #                         # Convert outputs probabilities to predicted class (0 or 1)
        #                         predicted = (outputs > 0.2).float()
        #                         print(predicted,gt)
        #                         # print(outputs.cpu().numpy(), predicted.cpu().numpy(), gt.cpu().numpy())
        #                         gt_positive = gt[0] == 1
        #                         gt_positive = gt_positive.float()

        #                         # Update total and correct counts
        #                         total += gt_positive.sum().item()
        #                         correct += (predicted[0][gt_positive.bool()] == gt[0][gt_positive.bool()]).sum().item()
        

        #         if total > 0:
        #             accuracy = correct / total
        #             print(f'Accuracy for gt=1: {accuracy}')
        #         else:
        #             print('No positive gt in this batch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='/home/notebook/data/personal/USS00063/prepare_data/', help=' data_dir')
    parser.add_argument('-model_name', type=str, default='VSegv3', help='GIANet,UNet_dummy1,BRDNet')
    parser.add_argument('-fp16', action='store_true', default=False)

    parser.add_argument('-batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-epochs', type=int, default=1500, help='epochs')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-r', type=float, default=0.18, help='r')
    parser.add_argument('-resume', type=int, default=0, help='resume')
    parser.add_argument('-resume_path', type=str, default=False, help='resume_path')
    parser.add_argument('-return_loss', action='store_true', help='#nep',default=True)
    parser.add_argument('-frames', type=int, default=3, help='frames as denoising input')
    parser.add_argument('-test_interval', type=int, default=1, help='epoch')
    parser.add_argument('-save_interval', type=int, default=50, help='epoch')
    parser.add_argument('-need_#nep', action='store_true', help='#nep',default=False)
    parser.add_argument('-unpack', action='store_true', help='unpack',default=False)
    parser.add_argument('-selected_device', type=int, default=0, help='selected_device')
    parser.add_argument('-size', type=int, default=256, help='width')
    parser.add_argument('-save_dir', type=str, default='./saved_model')
    parser.add_argument('-scheduler', type=str, default='CosineAnnealingLR', help = 'ReduceLROnPlateau CosineAnnealingLR')
    parser.add_argument('-debug', action='store_true', help='debug')

    parser.add_argument('-video',  action='store_true',default=True)
    parser.add_argument('-parallel', action='store_true', help='parallel',default=False)
    parser.add_argument('-prefix', type=str, default='full_training_set')
    parser.add_argument('-edge_loss', action='store_true', help='edge_loss',default=False)
    parser.add_argument('-opt', type=str, required=False, default='options/vrt/008_train_vrt_videodenoising_davis.json')
    parser.add_argument('-sigma', action='store_true', default=False)
    parser.add_argument('-Is_Image', action='store_true', default=False)
    parser.add_argument('-three2one', action='store_true', default=False)
    parser.add_argument('-use_hflip', type=bool, default=True, help='use_hflip')
    parser.add_argument('-use_rot', type=bool, default=True, help='use_rot')



    args = parser.parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            dist.destroy_process_group()  
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")