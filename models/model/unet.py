from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer
from models.model.scene_model import create_scene_model
from models.model.text_model import BERTEmbedder
from models.base import MODEL
import numpy as np
import matplotlib.pyplot as plt
from models.model.embodiment_attention import SpatialEncodingTransformerEncoderLayer, EmbodimentTransformerEncoder

@MODEL.register()
class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()
        self.collected_points = []
        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x
        self.use_llm = cfg.use_llm
        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        if self.use_llm:
            self.text_model = BERTEmbedder(n_embed=512, n_layer=32)
        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.h_dim = 384 + 32
        self.padding_dim = 96
        # self.robot_dim = 32

        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.h_dim, 1)
        )
        self.padding_embed = nn.Sequential(
            nn.Linear(27, 64),
            nn.SiLU(),
            nn.Linear(64, self.padding_dim),  # -> 96
        )

        # Graphormer
        self.graph_dim = 2 + 3 + 3 + 3
        self.input_layer_norm = nn.LayerNorm([self.graph_dim])
        self.graph_embedding = nn.Linear(self.graph_dim, 512)
        self.positional_encoding = nn.Identity()
        self.positional_encoding_dropout = nn.Dropout(p=0.1)

        self.feedforward_dim = 1024
        self.num_attention_heads = 16
        self.num_layers = 1
        encoder_layer_kwargs = {
            'd_model': 512,
            'nhead': self.num_attention_heads,
            'dim_feedforward': self.feedforward_dim,
            'norm_first': True, # found to have much more stable training performance in recent literature,
            'dropout': 0.1
        }
        encoder_kwargs = {
            'num_layers': self.num_layers
        }
        graph_encoder_layer = SpatialEncodingTransformerEncoderLayer(**encoder_layer_kwargs)
        self.graph_encoder = EmbodimentTransformerEncoder(graph_encoder_layer, **encoder_kwargs)

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model,
                    self.transformer_num_heads,
                    self.transformer_dim_head,
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor, padding_mask: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature

        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
            # robot_class_flag = robot_class_flag.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb) # [B, 1] -> [B, 1024]

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L> [B, 27] -> [B, 384+32]

        ## robot embedding
        # robot_emb = self.robot_embed(robot_class_flag.float())  # [B, 1] -> [B, 32]
        # robot_emb = robot_emb.unsqueeze(-1)
        ## padding embedding
        padding_emb = self.padding_embed(padding_mask)  # [B, 27] -> [B, 96]
        padding_emb = padding_emb.unsqueeze(-1)

        h = torch.cat([h, padding_emb], dim=1)  # [B, 512, 1]

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        # h: [B, 512, 1]
        # t_emb: [B, 1024, 1]
        # cond: [B, 8, 512]
        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond, graph_feat=graph_feat)

        # h: [B, 512]
        h = self.out_layers(h) # [B, 512] -> [B, 27]
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def graph_encode(self, data: Dict) -> torch.Tensor: # TODO: more properties

        adjacency_matrix = data['adjacency_matrix'] # B, 27, 27
        parent_matrix = data['parent_matrix'] # B, 27, 27
        child_matrix = data['child_matrix'] # B, 27, 27

        padding_mask = data['padding_mask']

        joint_limits = data['joint_limits'] # B, 27, 2
        joint_origin = data['joint_origin'] # B, 27, 3
        joint_axis = data['joint_axis'] # B, 27, 3
        joint_box = data['joint_box']
        joint_tokens = torch.cat([joint_limits, joint_origin, joint_axis, joint_box], dim=-1)
        joint_tokens = self.input_layer_norm(joint_tokens)
        joint_tokens = self.graph_embedding(joint_tokens)
        joint_tokens = self.positional_encoding(joint_tokens)
        joint_tokens = self.positional_encoding_dropout(joint_tokens)

        joint_tokens = self.graph_encoder(joint_tokens, padding_mask, adjacency_matrix, parent_matrix, child_matrix)

        return joint_tokens

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
            if self.use_llm:
                text_embedding = self.text_model(data['text'])[:,0,:]   # Get [CLS] embedding (all x 77 x 512) -> (all x 512)
                batch_embedding = torch.split(text_embedding, data['sentence_cnt']) # tuple of splitted embedding (? x 512)
                text_feat = torch.cat([a.max(dim=0,keepdim=True)[0] for a in batch_embedding], dim=0).reshape(b,1,-1) # (b,1,512)
                scene_feat = torch.cat([text_feat, scene_feat], dim=1)

        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        elif self.scene_model_name == 'PointNet2':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat
