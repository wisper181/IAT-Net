import torch
import torch.nn as nn
import sys
import os
import numpy as np

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from .backbone_module import Pointnet2Backbone
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead
from .image_feature_module import ImageFeatureModule, ImageMLPModule, append_img_feat
from .attention_module import IA_Layer, Atten_Fusion_Conv

def sample_valid_seeds(mask, num_sampled_seed=1024):
    """
    (TODO) write doc for this function
    """
    mask = mask.cpu().detach().numpy() # B,N
    all_inds = np.arange(mask.shape[1]) # 0,1,,,,N-1
    batch_size = mask.shape[0]
    sample_inds = np.zeros((batch_size, num_sampled_seed))
    for bidx in range(batch_size):
        valid_inds = np.nonzero(mask[bidx,:])[0] # return index of non zero elements
        if len(valid_inds) < num_sampled_seed:
            assert(num_sampled_seed <= 1024)
            rand_inds = np.random.choice(list(set(np.arange(1024))-set(np.mod(valid_inds, 1024))),
                                        num_sampled_seed-len(valid_inds),
                                        replace=False)
            cur_sample_inds = np.concatenate((valid_inds, rand_inds))
        else:
            cur_sample_inds = np.random.choice(valid_inds, num_sampled_seed, replace=False)
        sample_inds[bidx,:] = cur_sample_inds
    sample_inds = torch.from_numpy(sample_inds).long()
    return sample_inds



class IATDetector(nn.Module):
    r"""
        Image Attention Transformer Network for 3D Object Detection

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 max_imvote_per_pixel=3, image_feature_dim=18, image_hidden_dim=128, pc_feature_dim=288, joint_feature=288 + 128
                 ):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.image_feature_dim = image_feature_dim
        self.image_hidden_dim = image_hidden_dim
        self.pc_feature_dim = pc_feature_dim

        self.joint_feature = joint_feature
        self.joint_feature = self.image_hidden_dim + self.pc_feature_dim

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, width=self.width, pc_feature_dim=self.pc_feature_dim)

        # Image feature extractor
        self.image_feature_extractor = ImageFeatureModule(max_imvote_per_pixel=self.max_imvote_per_pixel)
        # MLP on image features before fusing with point features
        self.image_mlp = ImageMLPModule(image_feature_dim, image_hidden_dim=image_hidden_dim)

        self.final_fusion_img_point = Atten_Fusion_Conv(image_hidden_dim,
                                                        pc_feature_dim,
                                                        image_hidden_dim + pc_feature_dim)


        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(self.pc_feature_dim)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                         mean_size_arr, num_proposal, self.joint_feature)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(self.joint_feature, self.joint_feature, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(self.joint_feature, self.joint_feature, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, self.joint_feature))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, self.joint_feature))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, self.joint_feature))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    self.joint_feature, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                     mean_size_arr, num_proposal, self.joint_feature))

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        end_points.update(inputs)

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        # add img features
        img_feat_list = self.image_feature_extractor(end_points)
        assert len(img_feat_list) == self.max_imvote_per_pixel
        xyz, features, seed_inds = append_img_feat(img_feat_list, end_points)
        seed_sample_inds = sample_valid_seeds(features[:, -1, :], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1, features.shape[1], 1))
        xyz = torch.gather(xyz, 1, seed_sample_inds.unsqueeze(-1).repeat(1, 1, 3))
        seed_inds = torch.gather(seed_inds, 1, seed_sample_inds)

        pc_features = features[:, :self.pc_feature_dim, :].contiguous()
        img_features = features[:, self.pc_feature_dim:, :].contiguous()

        img_features = self.image_mlp(img_features)
        joint_features = self.final_fusion_img_point(pc_features, img_features)



        # # Query Points Generation
        points_xyz = xyz
        points_features = joint_features
        end_points['seed_inds'] = seed_inds
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = joint_features

        features = end_points['seed_features']


        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, pc_features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(pc_features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal

        else:
            raise NotImplementedError

        # Proposal
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=end_points,
                                                            prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                           base_xyz=cluster_xyz,
                                                           end_points=end_points,
                                                           prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
