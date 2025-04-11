from functools import partial

import timm
from transformers import AutoModel, RobertaModel

from models.losses import CLIP_Loss, CyCLIP_Loss, SogCLR_Loss, iSogCLR_Loss, iSogCLR_TempNet_Loss, VICReg_Loss, onlineCLR_Loss
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformer import VisionTransformer, TextTransformer, Transformer
import numpy as np


class CLIP(nn.Module):
    def __init__(self,           
                 image_encoder = None,
                 text_encoder = None,
                 embed_dim = 256,
                 init_model = True,
                 world_size = 8,
                 ita_type = 'clip',
                 sogclr_gamma = 0.9,
                 rho = 8.0,
                 tau_init = 0.01,
                 temp = 0.01,
                 learnable_temp = False,
                 personalized_tau = False,
                 vicreg_sim_coeff = 25.0, 
                 vicreg_std_coeff = 25.0,
                 N = 10000,
                 proto_std = 10.0,
                 proto_num = 256,
                 upper_rho_plus = 0.0,
                 proto_weight = 1.0,
                 sinkhorn_eps = 0.05,
                 swav_temp = 0.1,
                 swav_weight = 1.0,
                 batch_size_curr = 256
                 ):
        super().__init__()

        self.temp = temp
        self.learnable_temp = learnable_temp
        self.personalized_tau = personalized_tau
        self.bsz = batch_size_curr
        self.is_text_transformer = None  # flag to check whether we have to go with text transformer
        ## Temporary Temperature tensor for evaluation ##
        nonscalar_logit_scale = False
        lshape = [1] if nonscalar_logit_scale else []
        #self.logit_scale = nn.Parameter(torch.ones(lshape) * np.log(1 / 0.07)) ## causing DDP error
        self.register_buffer("logit_scale", torch.ones(lshape) * np.log(1 / 0.07))


        if self.learnable_temp:
            if not personalized_tau:
                self.temp = nn.Parameter(torch.ones([]) * self.temp)
            else:
                self.image_temp = nn.Parameter(torch.ones(N) * self.temp)
                self.text_temp = nn.Parameter(torch.ones(N) * self.temp)

        model = None
        if image_encoder == 'resnet50':
            cached_model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
            model = timm.create_model(image_encoder, pretrained=False)
            model.load_state_dict(torch.load(cached_model_path))
            self.visual_encoder = model
            #self.visual_encoder = timm.create_model(image_encoder, pretrained=init_model)
            self.visual_encoder.reset_classifier(0)
            self.vision_proj = nn.Linear(self.visual_encoder.num_features, embed_dim)   
        else:
            model = VisionTransformer(
                image_size=224,        
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                mlp_ratio=4.0,
                ls_init_value = None,
                output_tokens=False   
            )
            print('we have created the Vision Transformer Base 32')
            self.visual_encoder = model
            # we do not project now after vision encoder
            #self.vision_proj = nn.Linear(512, embed_dim)   

        if text_encoder == 'roberta-large':
            self.text_encoder = RobertaModel.from_pretrained(text_encoder, local_files_only=True)
            self.text_proj = nn.Linear(1024, embed_dim)
        elif text_encoder == 'Transformer':
            self.text_encoder = TextTransformer(
                context_length=77,  
                vocab_size=49408,  
                width=512,          
                heads=8,           
                layers=12,          
                mlp_ratio=4.0,      
                ls_init_value=None, 
                output_dim=512,    
                embed_cls=False,    
                no_causal_mask=False,  
                pad_id=0,           
                pool_type='argmax', 
                proj_type='linear',  
                proj_bias=False,    
                output_tokens=False  
            )
            print('we have created the Text Transformer')
            self.is_text_transformer = True
            # we do not need the Projection after Text Transformer 
            #self.text_proj = nn.Linear(768, embed_dim)
        else:
            self.text_encoder = AutoModel.from_pretrained(text_encoder, local_files_only=True)
            self.text_proj = nn.Linear(768, embed_dim)
            self.is_text_transformer = False

        if not init_model:
            self.text_encoder.init_weights()

        self.ita_type = ita_type

        if self.ita_type == 'clip':
            if not personalized_tau:
                self.criterion = CLIP_Loss(world_size=world_size, personalized_tau=personalized_tau, temperature=self.temp)
            else:
                self.criterion = CLIP_Loss(world_size=world_size, personalized_tau=personalized_tau, image_tau=self.image_temp, text_tau=self.text_temp)

        elif self.ita_type == 'cyclip':
            self.criterion = CyCLIP_Loss(world_size=world_size, temperature=self.temp)

        elif self.ita_type == 'vicreg':
            self.criterion = VICReg_Loss(world_size=world_size, dim_size=embed_dim, sim_coeff=vicreg_sim_coeff, std_coeff=vicreg_std_coeff)

        elif self.ita_type == 'sogclr':
            self.criterion = SogCLR_Loss(N=N, world_size=world_size, gamma=sogclr_gamma, temperature=self.temp)

        elif self.ita_type == 'isogclr':
            self.criterion = iSogCLR_Loss(N=N, world_size=world_size, gamma=sogclr_gamma, rho=rho)

        elif self.ita_type == 'onlineclr':
            self.criterion = onlineCLR_Loss(world_size=world_size, temperature=self.temp, gamma=sogclr_gamma)

        elif self.ita_type == 'isogclr_tempnet': # only use tempnet with new derivation, more deep structures
            self.criterion = iSogCLR_TempNet_Loss(N=N, world_size=world_size, gamma=sogclr_gamma, rho=rho, feature_dim=embed_dim, bsz = self.bsz)

        else:
            raise NotImplementedError


    def forward(self, image, text, idx, text_idx, epoch, max_epoch, return_feat=False):
        # gamma decay
        self.criterion.adjust_hyperparams(epoch)

        if self.learnable_temp:
            with torch.no_grad():
                if not self.personalized_tau:
                    self.temp.clamp_(0.001, 0.5)
                else:
                    self.image_temp.clamp_(0.001, 0.5)
                    self.text_temp.clamp_(0.001, 0.5)

        #with torch.no_grad():
        image_embeds = self.visual_encoder(image)
        #image_embeds = self.vision_proj(image_embeds)
        image_feat = F.normalize(image_embeds, dim=-1) 

        if self.is_text_transformer:
            #text_output = self.text_encoder(text.input_ids)
            text_output = self.text_encoder(text)
            #text_embeds = self.text_proj(text_output)
            text_embeds = text_output
        else:
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, output_hidden_states=False)
            text_embeds = self.text_proj(text_output.last_hidden_state[:,0,:])
        text_feat = F.normalize(text_embeds, dim=-1)   

        if return_feat:
            return image_feat, text_feat         

        avg_image_tau = None
        avg_text_tau = None

        info_dict = {}

        if self.ita_type in ['clip', 'cyclip']:
            if self.personalized_tau:
                image_ids = concat_all_gather(idx)
                text_ids = concat_all_gather(text_idx)
                loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids)

            else:
                loss_ita = self.criterion(image_feat, text_feat)

        elif self.ita_type == 'vicreg':
            loss_ita = self.criterion(image_embeds, text_embeds)

        elif self.ita_type == 'sogclr':
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch)

        elif self.ita_type == 'isogclr':
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita, image_tau, text_tau = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'image_tau':image_tau, 'text_tau':text_tau, 'image_ids': image_ids.cpu().numpy(), 'text_ids': text_ids.cpu().numpy()}

        elif self.ita_type == 'isogclr_tempnet':
            image_ids = concat_all_gather(idx)
            text_ids = concat_all_gather(text_idx)
            loss_ita, image_tau, text_tau, scores = self.criterion(image_feat, text_feat, image_ids, text_ids, epoch, max_epoch)
            info_dict = {'image_tau':image_tau, 'text_tau':text_tau, 'image_ids': image_ids.cpu().numpy(), 'text_ids': text_ids.cpu().numpy(), 'hardness_aware_score':scores}

        elif self.ita_type == 'onlineclr':
            loss_ita = self.criterion(image_feat, text_feat)

        else:
            raise NotImplementedError

        return loss_ita, info_dict



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output        

