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
from image_feature_module import ImageFeatureModule, ImageMLPModule, append_img_feat

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



#================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        # print(channels) # [128, 288]
        rc = self.pc // 4

        # nn.Sequential 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        # self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),  # nn.conv1d(in_channels, outchannel, kernel_size, stride=1,padding=0,dilation=1,groups=1)
        #                             nn.BatchNorm1d(self.pc),
        #                             nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)   #nn.Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas_f.shape, point_feas_f.shape) # torch.Size([1024, 128]) torch.Size([1024, 288])

        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # print(ri.shape, rp.shape)   # torch.Size([1024, 72]) torch.Size([1024, 72])

        att = torch.sigmoid(self.fc3(torch.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(att.shape)    # torch.Size([1, 1, 1024])
        # print(img_feas.size(), att.size())

        # img_feas_new = self.conv1(img_feas)
        # # print(img_feas_new.shape)   # torch.Size([1, 288, 1024])
        # out = img_feas_new * att

        # # img_feas_new = self.conv1(img_feas)
        # # print(img_feas_new.shape)   # torch.Size([1, 288, 1024])
        out = img_feas * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        # print('inplanes_I:', inplanes_I)
        # print('inplanes_P:', inplanes_P)
        # print('outplanes:', outplanes)
        # channels = [inplanes_I, inplanes_P]
        # print('channels:', channels)
        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        # self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        # self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape) # torch.Size([1, 288, 1024]) torch.Size([1, 128, 1024])

        img_features =  self.IA_Layer(img_features, point_features)
        # print("img_features:", img_features.shape)  # img_features: torch.Size([1, 288, 1024])

        #fusion_features = img_features + point_features
        fusion_features = torch.add([point_features, img_features], dim=1)
        # fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features



class GroupFreeDetector(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

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
                 max_imvote_per_pixel=3, image_feature_dim=18, image_hidden_dim=288, pc_feature_dim=288, joint_feature=288
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
        # self.joint_feature = self.pc_feature_dim * 2
        # self.joint_feature = self.image_hidden_dim + self.pc_feature_dim

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
            # self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                         mean_size_arr, num_proposal, self.joint_feature)
        # self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
        #                                  mean_size_arr, num_proposal, 288)
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

        # self.write2file()

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

        # print('end_points[fp2_xyz]', end_points['fp2_xyz'].size())
        # print('end_points[fp2_xyz]', end_points['fp2_xyz'])
        # print('end_points[fp2_features]', end_points['fp2_features'].size())
        # print('end_points[fp2_features]', end_points['fp2_features'])
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/demo_file/endpoints_fp2_xyz' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['fp2_xyz'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/demo_file/endpoints_fp2_features' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['fp2_features'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)

        # # Query Points Generation
        # points_xyz = end_points['fp2_xyz']
        # points_features = end_points['fp2_features']
        # xyz = end_points['fp2_xyz']
        # features = end_points['fp2_features']
        # end_points['seed_inds'] = end_points['fp2_inds']
        # end_points['seed_xyz'] = xyz
        # end_points['seed_features'] = features

        # add img features
        img_feat_list = self.image_feature_extractor(end_points)
        assert len(img_feat_list) == self.max_imvote_per_pixel
        xyz, features, seed_inds = append_img_feat(img_feat_list, end_points)
        seed_sample_inds = sample_valid_seeds(features[:, -1, :], 1024).cuda()
        features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1, features.shape[1], 1))
        xyz = torch.gather(xyz, 1, seed_sample_inds.unsqueeze(-1).repeat(1, 1, 3))
        seed_inds = torch.gather(seed_inds, 1, seed_sample_inds)

        # print('seed_inds:', seed_inds.size())
        # print('seed_inds:', seed_inds)
        # print('fp2_inds:', end_points['fp2_inds'].size())
        # print('fp2_inds:', end_points['fp2_inds'])
            # fp2_inds: torch.Size([3, 1024])
            # fp2_inds: tensor([[0, 8572, 905, ..., 13480, 1389, 3235],
            #                   [0, 5971, 18998, ..., 1786, 17642, 6887],
            #                   [0, 9684, 7220, ..., 13376, 7712, 7107]], device='cuda:0',
            #                  dtype=torch.int32)

        # pc_features = features[:, :256, :]
        # img_features = features[:, 256:, :]

        pc_features = features[:, :self.pc_feature_dim, :].contiguous()
        img_features = features[:, self.pc_feature_dim:, :].contiguous()

        # print('img_features:', img_features.size())
        # print('img_features:',img_features[:,:,0:3])
        img_features = self.image_mlp(img_features)
        # print('img_features:', img_features.size())
        # joint_features = torch.cat((pc_features, img_features), 1)
        # print('joint_features:', joint_features.size())
        # print('pc_features:', pc_features.shape)
        # print('pc_features:', pc_features)
        # print('img_features:', img_features.shape)
        # print('img_features:', img_features)
        joint_features = self.final_fusion_img_point(pc_features, img_features)



        # # Query Points Generation
        # points_xyz = end_points['fp2_xyz']
        # points_features = end_points['fp2_features']
        # xyz = end_points['fp2_xyz']
        # features = end_points['fp2_features']
        # end_points['seed_inds'] = end_points['fp2_inds']
        # end_points['seed_xyz'] = xyz
        # end_points['seed_features'] = features

        # # Query Points Generation
        points_xyz = xyz
        points_features = joint_features
        end_points['seed_inds'] = seed_inds
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = joint_features

        features = end_points['seed_features']

        # print('end_points[seed_xyz]', end_points['seed_xyz'].size())
        # print('end_points[seed_xyz]', end_points['seed_xyz'])
        # print('end_points[seed_features]', end_points['seed_features'].size())
        # print('end_points[seed_features]', end_points['seed_features'])
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/demo_file/endpoints_seed_xyz' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['seed_xyz'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/demo_file/endpoints_seed_features' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['seed_features'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        # print('end_points[seed_features]:', end_points['seed_features'].size())   #[3, 544, 1024]
        # print('end_points[seed_features]:', end_points['seed_features'])
        #     end_points[seed_features]: torch.Size([3, 544, 1024])
        #     end_points[seed_features]: tensor([[[0.7337, 0.2099, 0.2956, ..., 0.1095, 0.4125, 0.3232],
        #                                     [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
        #                                     [0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000],
        #                                     ...,
        #                                     [0.0000, 0.1715, 0.0318, ..., 0.0000, 0.1530, 0.0223],

        # print('end_points[seed_xyz]:', end_points['seed_xyz'].size())   # end_points[seed_xyz]: torch.Size([3, 1024, 3])
        # print('end_points[seed_xyz]:', end_points['seed_xyz'][:, :10, :3])
        # print('end_points[seed_features]:', end_points['seed_features'].size())  # end_points[seed_features]: torch.Size([3, 288, 1024])
        # print('end_points[seed_features]:', end_points['seed_features'][:, :10, :3])
        # print('end_points[seed_inds]:', end_points['seed_inds'].size())   # end_points[seed_inds]: torch.Size([3, 1024])
        # print('end_points[seed_inds]:', end_points['seed_inds'][:, :10])
        # # torch.save(end_points['seed_xyz'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoints_seed_xyz.pth')
        # # torch.save(end_points['seed_features'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoints_seed_features.pth')
        # # torch.save(end_points['seed_inds'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_seed_inds.pth')
        # # write2file(endpoints_seed_xyz, end_points['seed_xyz'])
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/endpoints_seed_xyz' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['seed_xyz'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/endpoints_seed_features' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # temp = torch.transpose(end_points['seed_features'], 1, 2)
        # # 将tensor变量转化为numpy类型
        # x = temp.cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_seed_inds' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['seed_inds'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)

        #print
        # end_points[seed_xyz]: torch.Size([3, 1024, 3])
        # end_points[seed_xyz]: tensor([[[-0.7976, 1.8609, -0.1991],
        #                                [3.7762, 8.0546, 2.5170],
        #                                [3.2067, 4.9265, -1.1988],
        #                                [-0.4339, 7.2174, 2.5035],
        #                                [-0.7089, 5.6775, -1.1271],
        #                                [1.4737, 5.3039, 1.2141],
        #                                [1.6494, 2.6214, -1.2620],
        #                                [-1.0783, 4.2870, 1.1806],
        #                                [1.7941, 7.0488, 2.5338],
        #                                [-0.2159, 3.6304, -1.1839]],
        #
        #                               [[-0.2847, 1.3172, -0.7087],
        #                                [2.7249, 7.3262, 0.9631],
        #                                [-2.4730, 4.2580, 0.2912],
        #                                [-0.4634, 7.1759, 2.4733],
        #                                [1.8638, 3.1804, 1.1718],
        #                                [1.8785, 2.6446, -1.3476],
        #                                [2.8724, 5.0767, 0.0096],
        #                                [-0.4073, 3.6990, 1.2346],
        #                                [1.5458, 6.8279, 2.4664],
        #                                [-1.0628, 3.0926, -0.5097]],
        #
        #                               [[1.0444, 2.7606, -1.2892],
        #                                [-1.9395, 7.9927, 2.5259],
        #                                [3.1658, 7.9041, 2.3637],
        #                                [-2.7236, 4.3356, 0.3229],
        #                                [2.7591, 4.3094, 1.1101],
        #                                [0.4051, 5.9029, 2.4365],
        #                                [-1.5676, 2.4843, -1.1761],
        #                                [-0.0102, 4.3716, 0.4247],
        #                                [3.4164, 6.3255, 0.1193],
        #                                [0.7583, 8.1321, 2.4374]]], device='cuda:0')
        # end_points[seed_features]: torch.Size([3, 288, 1024])
        # end_points[seed_features]: tensor([[[2.4630, 0.1930, 2.1985],
        #                                     [3.3399, 0.8604, 2.8096],
        #                                     [3.4362, 0.5911, 2.8927],
        #                                     [2.7639, 0.6867, 2.2830],
        #                                     [0.0000, 0.4191, 0.0479],
        #                                     [0.0000, 0.0000, 0.0000],
        #                                     [0.0000, 0.1422, 0.0000],
        #                                     [0.7199, 0.3087, 0.9416],
        #                                     [1.6207, 0.3200, 1.3059],
        #                                     [1.7592, 0.0000, 1.5868]],
        #
        #                                    [[3.1780, 1.1639, 2.7515],
        #                                     [2.9862, 2.2448, 3.0964],
        #                                     [3.7181, 2.3462, 3.3185],
        #                                     [2.6179, 1.9168, 2.3196],
        #                                     [0.2713, 0.6041, 0.0000],
        #                                     [0.0000, 0.0000, 0.0000],
        #                                     [0.0000, 0.2213, 0.0000],
        #                                     [1.0880, 0.7672, 0.9973],
        #                                     [1.8977, 1.5281, 1.5890],
        #                                     [2.2524, 1.0567, 1.9169]],
        #
        #                                    [[3.3873, 0.4806, 1.9379],
        #                                     [2.7092, 1.1624, 1.4578],
        #                                     [3.2818, 0.9272, 1.9809],
        #                                     [2.3030, 0.8330, 1.8069],
        #                                     [0.6274, 0.7073, 0.4922],
        #                                     [0.0000, 0.0000, 0.0000],
        #                                     [0.0000, 0.1549, 0.0000],
        #                                     [1.0978, 0.5217, 0.7521],
        #                                     [2.1469, 0.0823, 1.1051],
        #                                     [1.8235, 0.2310, 0.8721]]], device='cuda:0')
        # end_points[seed_inds]: torch.Size([3, 1024])
        # end_points[seed_inds]: tensor([[0, 13167, 16168, 6958, 1748, 6972, 9538, 691, 6088, 17756],
        #                                [0, 9672, 281, 9614, 606, 8143, 16431, 3226, 10780, 4163],
        #                                [0, 1474, 18514, 19933, 6223, 4463, 13917, 2702, 8711, 14923]],
        #                               device='cuda:0', dtype=torch.int32)


        # kps only select 256 xyz and features from 1024, no change
        if self.sampling == 'fps':
            # xyz, features, sample_inds = self.fps_module(xyz, features)
            xyz, features, sample_inds = self.fps_module(xyz, pc_features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            # points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            points_obj_cls_logits = self.points_obj_cls(pc_features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            # xyz, features, sample_inds = self.gsample_module(xyz, pc_features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            # print(xyz.shape)    # torch.Size([8, 256, 3])
            # print(features.shape)   # torch.Size([8, 576, 256])
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal

            # # features = torch.gather(features, -1, seed_sample_inds.unsqueeze(1).repeat(1, features.shape[1], 1))
            # img_features_sample = torch.gather(img_features, -1, sample_inds.unsqueeze(1).repeat(1, img_features.shape[1], 1).long())
            #
            # # print('img_features:', img_features.shape)
            # # print('img_features:', img_features)
            # # print('sample_inds:', sample_inds.shape)
            # # print('sample_inds:', sample_inds)
            # # print('img_features_sample:', img_features_sample.shape)  # img_features_sample: torch.Size([1, 128, 256])
            # # print('img_features_sample:', img_features_sample)
            # # print(img_features[:, :, sample_inds[0, 0].item()])
            # # print(img_features[:, :, sample_inds[0, 1].item()])
            # cluster_feature = torch.cat((features, img_features_sample), dim=1)
            # end_points['query_points_feature'] = cluster_feature
            # # print(end_points['query_points_feature'].shape) # torch.Size([1, 416, 256])

        else:
            raise NotImplementedError

        # print('end_points[query_points_xyz]:', end_points['query_points_xyz'].size()) # end_points[query_points_xyz]: torch.Size([3, 256, 3])
        # print('end_points[query_points_xyz]:', end_points['query_points_xyz'][:, :10, :3])
        # print('end_points[query_points_feature]:', end_points['query_points_feature'].size())  # end_points[query_points_feature]: torch.Size([3, 288, 256])
        # print('end_points[query_points_feature]:', end_points['query_points_feature'][:, :10, :3])
        # print('end_points[query_points_sample_inds]:', end_points['query_points_sample_inds'].size())  # end_points[query_points_sample_inds]: torch.Size([3, 256])
        # print('end_points[query_points_sample_inds]:', end_points['query_points_sample_inds'][:, :10])
        # # torch.save(end_points['query_points_xyz'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_query_points_xyz.pth')
        # # torch.save(end_points['query_points_feature'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_query_points_feature.pth')
        # # torch.save(end_points['query_points_sample_inds'], '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_query_points_sample_inds.pth')
        # # torch.save(end_points, '/home/tongYan/code/Group-Free-3D-master_demo/endpoint_query_points_sample_inds.pth')
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/query_points_xyz' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['query_points_xyz'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/query_points_feature' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        #
        # # end_points['xyz_pcl_normals'] = torch.transpose(end_points['xyz_pcl_normals'], 1, 2)
        # # end_points['seed_features'] = torch.cat((end_points['seed_features'], end_points['xyz_pcl_normals']), dim=1)
        # temp = torch.transpose(end_points['query_points_feature'], 1, 2)
        # # 将tensor变量转化为numpy类型
        # x = temp.cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        #
        # file_name = '/home/tongYan/code/Group-Free-3D-master_demo/query_points_sample_inds' + '.txt'
        # file_handle = open(file_name, mode='w')  # w 写入模式
        # # 将tensor变量转化为numpy类型
        # x = end_points['query_points_sample_inds'].cpu().numpy()
        # # 将numpy类型转化为list类型
        # x = x.tolist()
        # # 将list转化为string类型
        # strNums = [str(x_i) + '\n' for x_i in x]
        # str1 = ",".join(strNums)
        # # 将str类型数据存入本地文件1.txt中
        # file_handle.write(str1)
        # # # print
        # end_points[query_points_xyz]: torch.Size([3, 256, 3])
        # end_points[query_points_xyz]: tensor([[[-0.8261, 1.5559, -0.7479],
        #                                        [-0.8164, 2.3902, -0.8859],
        #                                        [-0.5048, 2.0319, -1.1975],
        #                                        [0.4661, 2.6337, -0.6377],
        #                                        [-0.7925, 1.6375, -0.6321],
        #                                        [-0.7223, 1.5409, -1.2049],
        #                                        [-0.7873, 2.2681, -1.0218],
        #                                        [-0.6079, 1.9327, -1.2039],
        #                                        [-0.6828, 1.6489, -1.2000],
        #                                        [-0.8421, 1.4989, -1.1984]],
        #
        #                                       [[1.6080, 2.7478, -1.1468],
        #                                        [1.4920, 2.7475, -1.0661],
        #                                        [1.5893, 2.8007, -1.0663],
        #                                        [1.4345, 2.7224, -0.9508],
        #                                        [0.4606, 3.5357, 0.4151],
        #                                        [1.5599, 2.7169, -0.9161],
        #                                        [1.6893, 2.7287, -0.9056],
        #                                        [1.4692, 2.6225, -1.2946],
        #                                        [1.0138, 2.6268, -1.2693],
        #                                        [1.5427, 2.7391, -1.2971]],
        #
        #                                       [[0.6707, 4.2071, -0.1952],
        #                                        [0.6426, 4.2404, -0.0315],
        #                                        [0.9252, 4.3037, -0.4640],
        #                                        [0.6777, 4.2358, 0.1022],
        #                                        [0.4494, 4.2469, -0.1399],
        #                                        [1.0028, 4.2879, 0.2129],
        #                                        [1.6821, 4.2745, 0.4451],
        #                                        [0.9819, 3.7806, -0.9684],
        #                                        [0.0075, 4.3681, 0.6460],
        #                                        [0.8433, 4.3032, -0.5756]]], device='cuda:0')
        # end_points[query_points_feature]: torch.Size([3, 288, 256])
        # end_points[query_points_feature]: tensor([[[2.8876, 3.0789, 3.0367],
        #                                            [3.0261, 2.8769, 2.9600],
        #                                            [3.3830, 3.5043, 3.3838],
        #                                            [2.7691, 2.5405, 2.4606],
        #                                            [0.1615, 0.4296, 0.3107],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [0.8481, 1.0440, 1.0349],
        #                                            [1.7686, 2.0216, 1.8700],
        #                                            [2.0348, 1.9712, 1.8255]],
        #
        #                                           [[3.0364, 3.0692, 2.8220],
        #                                            [3.2427, 3.2088, 3.1323],
        #                                            [3.6465, 3.6495, 3.7051],
        #                                            [2.6609, 2.6873, 2.5463],
        #                                            [0.3438, 0.3558, 0.2442],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [1.2658, 1.2343, 1.2122],
        #                                            [1.9234, 1.9922, 2.0989],
        #                                            [2.0949, 1.9676, 2.0413]],
        #
        #                                           [[3.0680, 2.8645, 2.9840],
        #                                            [2.9493, 2.8892, 3.1300],
        #                                            [3.7742, 3.7809, 3.6504],
        #                                            [2.7543, 2.6200, 2.5688],
        #                                            [0.2344, 0.2634, 0.3081],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [0.0000, 0.0000, 0.0000],
        #                                            [1.2968, 1.1122, 1.0830],
        #                                            [2.1647, 2.2247, 2.0617],
        #                                            [2.2030, 1.9654, 2.1370]]], device='cuda:0')
        # end_points[query_points_sample_inds]: torch.Size([3, 256])
        # end_points[query_points_sample_inds]: tensor([[1015, 185, 782, 233, 899, 1020, 502, 278, 364, 25],
        #                                               [100, 486, 903, 624, 371, 226, 840, 357, 295, 533],
        #                                               [392, 798, 909, 455, 810, 794, 694, 112, 401, 615]],
        #                                              device='cuda:0', dtype=torch.int32)

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

    # def write2file(self, file_name, tensor_num):
    #     file_name = file_name + '.txt'
    #     file_handle = open(file_name, mode='w')  # w 写入模式
    #     # 将tensor变量转化为numpy类型
    #     x = tensor_num.cpu().numpy()
    #     # 将numpy类型转化为list类型
    #     x = x.tolist()
    #     # 将list转化为string类型
    #     strNums = [str(x_i) for x_i in x]
    #     str1 = ",".join(strNums)
    #     # 将str类型数据存入本地文件1.txt中
    #     file_handle.write(str1)