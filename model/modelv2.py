

import torch
from torch.cuda.amp import autocast as autocast
from einops import rearrange
import torch.nn as nn
import numpy as np
from torchvision.ops import sigmoid_focal_loss
try:
    from .blip2 import Blip2Base, disabled_train
    from .eva_vit import Trans_Block
    from .RMT import RecurrentMemoryTransformer
except:
    from blip2 import Blip2Base, disabled_train
    from eva_vit import Trans_Block
    from RMT import RecurrentMemoryTransformer


class VSegv2(Blip2Base):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()


        self.low_resource = config.low_resource

        self.vit_precision = config.vit_precision
        print(f'Loading VIT. Use fp16: {config.vit_precision}')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            config.vit_model, config.img_size, config.drop_path_rate, 
            config.use_grad_checkpoint, config.vit_precision, config.vit_model_path,
            temporal_downsample=config.temporal_downsample,
            no_lmhra=config.no_lmhra, 
            double_lmhra=config.double_lmhra,
            lmhra_reduction=config.lmhra_reduction, 
            gmhra_layers=config.gmhra_layers, 
            gmhra_drop_path_rate=config.gmhra_drop_path_rate,
            gmhra_dropout=config.gmhra_dropout, 
        )
        if config.freeze_vit:
            print("freeze vision encoder")
            if not config.freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)

            else:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
                
        # self.RMT = RecurrentMemoryTransformer(
        #     num_tokens = 512,               # number of tokens
        #     num_memory_tokens = 128,          # number of memory tokens, this will determine the bottleneck for information being passed to the future
        #     dim = 512,                        # model dimensions
        #     depth = 6,                        # transformer depth
        #     causal = True,                    # autoregressive or not
        #     dim_head = 64,                    # dimension per head
        #     heads = 8,                        # heads
        #     seq_len = 512*10,                   # sequence length of a segment
        #     use_flash_attn = True             # whether to use flash attention
        # )
        self.Frames = config.Frames
        self.blocks = nn.ModuleList([
                Trans_Block(dim = 512, num_heads = config.embedding_size//88, mlp_ratio= 4.3637)
                    for i in range(5)])
        self.norm = nn.LayerNorm(config.embedding_size)
        self.fc_norm = nn.LayerNorm(config.embedding_size)
        self.fn1 =nn.Linear(1408,512)  #feature number from 1408 to 512
        self.fn2 = nn.Linear(1286,512) #patch number from 1286 to 512
        
        # self.fn3 = nn.Linear(512*10,512)
        self.memory = None
        # self.head = nn.Linear(512, config.frames*2)
        self.head = nn.Linear(512, self.Frames)

        self.head_fc1 = nn.Linear(1024, self.Frames)
        self.head_fc2 = nn.Linear(512, 2)
        self.score_head = nn.Linear(config.embedding_size, 1)
        self.max_txt_len = config.max_txt_len
        self.cache_data = []
        self.relu = nn.ReLU()
        # weights = [1.0, 100]
    
    def head_classifier(self, x):
        
        x = rearrange(x, 'b t c -> b c t ')
        
        x = self.head_fc1(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.head_fc2(x)
        
        return x 
    
    
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def forward_all_features(self, interval_1,interval_2):

        with self.maybe_autocast():
            T = interval_1.shape[1]
            # use_image = True if T == 1 else False
            interval_1 = interval_1.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
            interval_2 = interval_2.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
            interval1_embeds = self.ln_vision(self.visual_encoder(interval_1))
            interval2_embeds = self.ln_vision(self.visual_encoder(interval_2))
        x = torch.concat((interval1_embeds, interval2_embeds), dim=1)
        return x

        
    def forward_feature(self, interval_1):

        # with self.maybe_autocast():
        T = interval_1.shape[1]
        # use_image = True if T == 1 else False
        interval_1 = interval_1.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        interval1_embeds = self.ln_vision(self.visual_encoder(interval_1))
        return interval1_embeds

    # def RMT_forward(self, x):

    #     #[1, 512, 512]
    #     RMT_output, self.memory, _ = self.RMT(x,self.memory)        # (1, 1024, 20000), (1, 128, 512), None
    #     return  RMT_output

    def clear_cache(self):
        self.cache_data = []
        self.memory = None
    

    
    def forward(self, interval_1,interval_2):
        
        vit_feature = self.forward_feature(interval_1)
        vit_feature = self.norm(vit_feature)

        vit_feature = self.fn1(vit_feature)
        vit_feature = rearrange(vit_feature, 'b c l -> b l c') # (patch number, feature vector) -> ( feature vector, patch number)
        vit_feature = self.fn2(vit_feature)                    # decrease the patch number from 1286 to 512
        out_vit_feature = rearrange(vit_feature, 'b l c -> b c l' ) # ( feature vector, patch number) -> (patch number, feature vector) 
          

        feature_vit_feature = self.forward_feature(interval_2)
        feature_vit_feature = self.norm(feature_vit_feature)
        feature_vit_feature = self.fn1(feature_vit_feature)
        feature_vit_feature = rearrange(feature_vit_feature, 'b c l -> b l c') # (patch number, feature vector) -> ( feature vector, patch number)
        feature_vit_feature = self.fn2(feature_vit_feature)                    # decrease the patch number from 1286 to 512
        feature_out_vit_feature = rearrange(feature_vit_feature, 'b l c -> b c l' ) # ( feature vector, patch number) -> (patch number, feature vector) 
          



        if len(self.cache_data) == 0:
            self.cache_data.append(out_vit_feature.detach().cpu())
            Prev_feature = torch.zeros_like(out_vit_feature)
        else:   
            Prev_feature = (self.cache_data.pop()).cuda()
            self.cache_data.append(out_vit_feature.detach().cpu())

        # RMT_feature = self.RMT_forward(out_vit_feature)
        # x = torch.cat((RMT_feature, Prev_feature, out_vit_feature), dim = 1)

        x = torch.cat(( Prev_feature, out_vit_feature,feature_out_vit_feature), dim = 1)
        # print('combined_feature: ', x.shape)
        for block in self.blocks:
            x = block(x)
        x = self.relu(x)
        x = x[:, 0]
        # x = self.head_classifier(x)
        x = self.head(x)
        # x = x.view(-1, self.Frames, 2)  # reshapes the output to (batch_size, 5, 2)
        # x = self.relu(x)
        x = torch.sigmoid(x)
        return x     
    
    
if __name__ == '__main__':
    import json
    from box import Box

    with open('/mnt/drive1/brick1/hsun/videoSeg/config/config.json', 'r') as f:
        config_dict = json.load(f)

    config = Box(config_dict)
    model = VSegv2(config.model).cuda()
    for i in range(3):
        video = torch.rand(1, 5, 3,  224, 224).cuda()

        print(video.shape)
        output = model(video,video)

        print(output.shape)