import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv, LayerNorm, GATConv

from HGCN_code import mae_utils
from HGCN_code.mae_utils import get_sinusoid_encoding_table#, Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from functools import partial
import pdb
import torch
import torch.nn as nn

from models.vision_transformer import PatchEmbed, Block, CBlock

from models.pos_embed import get_2d_sincos_pos_embed
from models.our import PreModel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)



class MaskedAutoencoderConvViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=512, depth=3, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, train_type_num=3):
        super().__init__()
        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
        self.patch_embed1 = PatchEmbed(in_chans=3, stride=1, embed_dim=3)
        self.patch_embed2 = PatchEmbed(in_chans=3, stride=1, embed_dim=3)
        self.patch_embed3 = PatchEmbed(in_chans=3, stride=1, embed_dim=3)

        self.patch_embed4 = nn.Linear(embed_dim, embed_dim)
        self.stage1_output_decode = nn.Conv1d(3, 1, 1, stride=1)
        self.stage2_output_decode = nn.Conv1d(3, 1, 1, stride=1)

        num_patches = train_type_num
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False)
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.blocks1 = nn.ModuleList([
            CBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks2 = nn.ModuleList([
            CBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # self.blocks3 = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim, bias=False)
        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,decoder_embed_dim,bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def forward_encoder(self, x, mask):

        x = self.patch_embed1(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        # B, _, C = x.shape
        # x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks1:
            x = blk(x)

        stage1_embed = self.stage1_output_decode(x).flatten(2)

        x = self.patch_embed2(x.squeeze(0))
        for blk in self.blocks2:
            x = blk(x)
        stage2_embed = self.stage2_output_decode(x).flatten(2)

        x = self.patch_embed3(x.squeeze(0))
        # x = x.flatten(2)#.permute(0, 2, 1)

        # x = self.patch_embed4(x)
        # # apply Transformer blocks
        # for blk in self.blocks3:
        #     x = blk(x)

        B, _, C = x.shape
        x = x[~mask].reshape(B, -1, C)  # ~mask means visible

        x = x + stage1_embed + stage2_embed
        x = self.norm(x)


        return x

    def forward_decoder(self, x, return_token_num):
        # embed tokens
        x = self.decoder_embed(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward(self, x, mask):
        x_vis = self.forward_encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, N, C = x_vis.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.forward_decoder(x_full, 0)  # [B, N_mask, 3 * 16 * 16]

        tmp_x = torch.zeros_like(x).to(device)
        Mask_n = 0
        Truth_n = 0
        for i, flag in enumerate(mask[0][0]):
            if flag:
                tmp_x[:, i] = x[:, pos_emd_vis.shape[1] + Mask_n]
                Mask_n += 1
            else:
                tmp_x[:, i] = x[:, Truth_n]
                Truth_n += 1
        return tmp_x


def Mix_mlp(dim1):
    return nn.Sequential(
        nn.Linear(dim1, dim1),
        nn.GELU(),
        nn.Linear(dim1, dim1))


class MixerBlock(nn.Module):
    def __init__(self, dim1, dim2):
        super(MixerBlock, self).__init__()

        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)

    def forward(self, x):
        x = x.transpose(0, 1)
        # z = nn.Linear(512, 3)(x)

        y = self.norm(x)
        # y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        y = y.transpose(0, 1)
        z = self.mix_mip_2(y)
        z = z.transpose(0, 1)
        x = x + z
        x = x.transpose(0, 1)

        # y = self.norm(x)
        # y = y.transpose(0,1)
        # y = self.mix_mip_1(y)
        # y = y.transpose(0,1)
        # x = self.norm(y)
        return x


def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ReLU(),
        nn.Dropout(p=dropout))


def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        #             GATConv(in_channels=dim1,out_channels=dim2),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


from torch_geometric.nn import GATConv,GCNConv,GENConv, GINConv,GMMConv,GPSConv, GINEConv, GATv2Conv, APPNP, GatedGraphConv, ARMAConv

class GINNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINNet, self).__init__()

        self.gin_conv = GINConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        ), eps=0.2, train_eps=True)

    def forward(self, x, edge_index):
        x = self.gin_conv(x, edge_index)
        return x

class GPSNet(nn.Module):
    def __init__(self, in_channels, out_channels, conv=None, heads=1, dropout=0.0, attn_dropout=0.0,
                 act='relu', act_kwargs=None, norm='batch_norm', norm_kwargs=None):
        super(GPSNet, self).__init__()

        self.gps_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True
        )

    def forward(self, x, edge_index):
        x = self.gps_conv(x, edge_index)
        return x

class GINNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim=None):
        super(GINNNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.gps_conv = GINEConv(self.mlp)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gps_conv(x, edge_index, edge_attr)
        return x

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphClassifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.appnp = APPNP(K=10, alpha=0.1)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.appnp(x, edge_index)
        return x

class GraphARMAConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stacks, num_layers):
        super(GraphARMAConv, self).__init__()

        self.armac = nn.ModuleList()
        self.armac.append(ARMAConv(input_dim, hidden_dim, num_layers))
        for _ in range(num_stacks - 1):
            self.armac.append(ARMAConv(hidden_dim, hidden_dim, num_layers))
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for armac in self.armac:
            x = armac(x, edge_index)
        x = self.fc(x)
        return x

# class GatedGraphClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GatedGraphClassifier, self).__init__()
#
#         self.ggc = GatedGraphConv(input_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x, edge_index):
#         x = self.ggc(x, edge_index)
#         x = self.fc(x)
#         return x
def GNN_relu_Block_new(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        # GATConv(in_channels=1024,out_channels=512),
        nn.Linear(1024, 512),
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))
class fusion_model_MaskedAutoencoderConvViT(nn.Module):
    def __init__(self, args, in_feats, n_hidden, out_classes, dropout=0.3, train_type_num=3):
        super(fusion_model_MaskedAutoencoderConvViT, self).__init__()

        # self.img_gnn_2 = SAGEConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GATConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GCNConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GENConv(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GINNet(in_channels=in_feats, out_channels=out_classes)
        # self.img_gnn_2 = GPSNet(in_channels=in_feats, out_channels=out_classes, conv=None)
        # self.img_gnn_2 = GATv2Conv(in_channels=in_feats, out_channels=out_classes)
        self.img_gnn_2 = GraphARMAConv(input_dim=in_feats, hidden_dim=256, output_dim=out_classes, num_stacks=2, num_layers=2)
        self.img_gnn_2 = PreModel(args, 2, 1024)
        self.img_relu_2 = GNN_relu_Block_new(out_classes)

        self.rna_gnn_2 = PreModel(args, 2, 1024)#SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.rna_relu_2 = GNN_relu_Block_new(out_classes)
        self.cli_gnn_2 = PreModel(args, 2, 1024)#SAGEConv(in_channels=in_feats, out_channels=out_classes)
        self.cli_relu_2 = GNN_relu_Block_new(out_classes)
        #         TransformerConv

        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_img = my_GlobalAttention(att_net_img)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_rna = my_GlobalAttention(att_net_rna)

        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(), nn.Linear(out_classes // 4, 1))
        self.mpool_cli = my_GlobalAttention(att_net_cli)

        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)

        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes // 4), nn.ReLU(),
                                      nn.Linear(out_classes // 4, 1))
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)

        # self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes,
        #                                      decoder_embed_dim=out_classes, encoder_depth=1, decoder_depth=1,
        #                                      train_type_num=train_type_num)
        self.mae = MaskedAutoencoderConvViT()

        self.mix = MixerBlock(train_type_num, out_classes)

        self.lin1_img = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_img = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_rna = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_rna = torch.nn.Linear(out_classes // 4, 1)
        self.lin1_cli = torch.nn.Linear(out_classes, out_classes // 4)
        self.lin2_cli = torch.nn.Linear(out_classes // 4, 1)

        self.norm_img = LayerNorm(out_classes // 4)
        self.norm_rna = LayerNorm(out_classes // 4)
        self.norm_cli = LayerNorm(out_classes // 4)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, all_thing, train_use_type=None, use_type=None, in_mask=[], mix=False):

        if len(in_mask) == 0:
            mask = np.array([[[False] * len(train_use_type)]])
        else:
            mask = in_mask

        data_type = use_type
        x_img = all_thing.x_img
        x_rna = all_thing.x_rna
        x_cli = all_thing.x_cli

        data_id = all_thing.data_id
        edge_index_img = all_thing.edge_index_image
        edge_index_rna = all_thing.edge_index_rna
        edge_index_cli = all_thing.edge_index_cli

        save_fea = {}
        fea_dict = {}
        num_img = len(x_img)
        num_rna = len(x_rna)
        num_cli = len(x_cli)

        att_2 = []
        pool_x = torch.empty((0)).to(device)
        if 'img' in data_type:
            loss_img,x_img = self.img_gnn_2(x_img, edge_index_img)
            x_img = self.img_relu_2(x_img)
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_2 = self.mpool_img(x_img, batch)
            att_2.append(att_img_2)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            loss_rna, x_rna = self.rna_gnn_2(x_rna, edge_index_rna)
            x_rna = self.rna_relu_2(x_rna)
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_2 = self.mpool_rna(x_rna, batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'cli' in data_type:
            loss_cli,x_cli = self.cli_gnn_2(x_cli, edge_index_cli)
            x_cli = self.cli_relu_2(x_cli)
            batch = torch.zeros(len(x_cli), dtype=torch.long).to(device)
            pool_x_cli, att_cli_2 = self.mpool_cli(x_cli, batch)
            att_2.append(att_cli_2)
            pool_x = torch.cat((pool_x, pool_x_cli), 0)

        fea_dict['mae_labels'] = pool_x

        if len(train_use_type) > 1:
            if use_type == train_use_type:
                mae_x = self.mae(pool_x, mask).squeeze(0)
                fea_dict['mae_out'] = mae_x
            else:
                k = 0
                tmp_x = torch.zeros((len(train_use_type), pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type), dtype=bool)
                for i, type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k += 1
                        mask[i] = False
                mask = np.expand_dims(mask, 0)
                mask = np.expand_dims(mask, 0)
                if k == 0:
                    mask = np.array([[[False] * len(train_use_type)]])
                mae_x = self.mae(tmp_x, mask).squeeze(0)
                fea_dict['mae_out'] = mae_x

            save_fea['after_mae'] = mae_x.cpu().detach().numpy()
            if mix:
                mae_x = self.mix(mae_x)
                save_fea['after_mix'] = mae_x.cpu().detach().numpy()

            k = 0
            if 'img' in train_use_type and 'img' in use_type:
                x_img = x_img + mae_x[train_use_type.index('img')]
                k += 1
            if 'rna' in train_use_type and 'rna' in use_type:
                x_rna = x_rna + mae_x[train_use_type.index('rna')]
                k += 1
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]
                k += 1

        att_3 = []
        pool_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            batch = torch.zeros(len(x_img), dtype=torch.long).to(device)
            pool_x_img, att_img_3 = self.mpool_img_2(x_img, batch)
            att_3.append(att_img_3)
            pool_x = torch.cat((pool_x, pool_x_img), 0)
        if 'rna' in data_type:
            batch = torch.zeros(len(x_rna), dtype=torch.long).to(device)
            pool_x_rna, att_rna_3 = self.mpool_rna_2(x_rna, batch)
            att_3.append(att_rna_3)
            pool_x = torch.cat((pool_x, pool_x_rna), 0)
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli), dtype=torch.long).to(device)
            pool_x_cli, att_cli_3 = self.mpool_cli_2(x_cli, batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x, pool_x_cli), 0)

        x = pool_x

        x = F.normalize(x, dim=1)
        fea = x

        k = 0
        if 'img' in data_type:
            fea_dict['img'] = fea[k]
            k += 1
        if 'rna' in data_type:
            fea_dict['rna'] = fea[k]
            k += 1
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
            k += 1

        k = 0
        multi_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            x_img = self.lin1_img(x[k])
            x_img = self.relu(x_img)
            x_img = self.norm_img(x_img)
            x_img = self.dropout(x_img)

            x_img = self.lin2_img(x_img).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_img), 0)
            k += 1
        if 'rna' in data_type:
            x_rna = self.lin1_rna(x[k])
            x_rna = self.relu(x_rna)
            x_rna = self.norm_rna(x_rna)
            x_rna = self.dropout(x_rna)

            x_rna = self.lin2_rna(x_rna).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_rna), 0)
            k += 1
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])
            x_cli = self.relu(x_cli)
            x_cli = self.norm_cli(x_cli)
            x_cli = self.dropout(x_cli)

            x_cli = self.lin2_rna(x_cli).unsqueeze(0)
            multi_x = torch.cat((multi_x, x_cli), 0)
            k += 1
        one_x = torch.mean(multi_x, dim=0)

        return (one_x, multi_x), save_fea, (att_2, att_3), fea_dict  










