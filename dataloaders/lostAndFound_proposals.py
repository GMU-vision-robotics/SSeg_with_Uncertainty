import numpy as np
import cv2
import json
import matplotlib.pyplot as plt 
import glob
from PIL import Image 
import os
import torch.utils.data as data
import torch
import torch.nn.functional as F

class LostAndFoundProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.rep_style = rep_style

		self.data_json_file = json.load(open('{}/{}_data_annotation.json'.format(self.dataset_dir, 'Lost_and_Found')))

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.class_names = ['unlabelled', 'road', 'building', \
							'pole', 'vegetation', 'sky', 'person', 'car', 'train', ]

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} images".format(len(self.data_json_file)))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals/lostAndFound'
		self.mask_ft_folder  = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features/lostAndFound'
		self.sseg_ft_folder  = '/projects/kosecka/yimeng/Datasets/Lost_and_Found/deeplab_ft_8_classes'

	def __len__(self):
		return len(self.data_json_file)

	def encode_segmap(self, mask):
		#merge ambiguous classes
		mask[mask == 6] = 7 # ground -> road
		mask[mask == 8] = 7 # sidewalk -> road
		mask[mask == 9] = 7 # parking -> road
		mask[mask == 22] = 21 # terrain -> vegetation
		mask[mask == 25] = 24 # rider -> person
		mask[mask == 32] = 24 # motorcycle -> person
		mask[mask == 33] = 24 # bicycle -> person
		mask[mask == 27] = 26 # truck -> car
		mask[mask == 28] = 26 # bus -> car
		mask[mask == 29] = 26 # caravan -> car
		mask[mask == 30] = 26 # trailer -> car
		mask[mask == 12] = 11 # wall -> building
		mask[mask == 13] = 11 # fence -> building
		mask[mask == 19] = 17 # traffic light -> pole
		mask[mask == 20] = 17 # traffic sign -> pole
		mask[mask == 18] = 17 # pole group -> pole

		# Put all void classes to zero
		for _voidc in self.void_classes:
			mask[mask == _voidc] = self.ignore_index
		for _validc in self.valid_classes:
			mask[mask == _validc] = self.class_map[_validc]
		return mask

	def get_num_proposal(self, i):
		v = self.data_json_file[str(i)]
		return len(v['regions'])

	def get_proposal(self, i, j=0):
		img_path = '{}/{}.png'.format(self.dataset_dir, i)
		lbl_path = '{}/{}_label.png'.format(self.dataset_dir, i)

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		index = np.array([j])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]) # B x 256 x 14 x 14

		#print('proposals.shape = {}'.format(proposals.shape))
		#print('mask_feature.shape = {}'.format(mask_feature.shape))
		#assert 1==2

		batch_sseg_feature = torch.zeros((1, 256, 14, 14))
		batch_sseg_label   = torch.zeros((1, 28, 28))

		x1, y1, x2, y2 = proposals[0]
		prop_x1 = int(max(round(x1), 0))
		prop_y1 = int(max(round(y1), 0))
		prop_x2 = int(min(round(x2), 2048-1))
		prop_y2 = int(min(round(y2), 1024-1))

		img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

		# sseg feature size is 1/8 of the original image size
		sseg_x1 = prop_x1//8
		sseg_y1 = prop_y1//8
		sseg_x2 = prop_x2//8
		sseg_y2 = prop_y2//8
		#print('sseg_x1 = {}, sseg_y1 = {}, sseg_x2 = {}, sseg_y2 = {}'.format(sseg_x1, sseg_y1, sseg_x2, sseg_y2))

		# rescale patch_sseg_feature to 14x14
		patch_sseg_feature = torch.tensor(sseg_feature[:, sseg_y1:sseg_y2, sseg_x1:sseg_x2]).unsqueeze(0) # 1 x 256 x prop_h/8 x prop_w/8
		#print('patch_sseg_feature.shape = {}'.format(patch_sseg_feature.shape))
		patch_sseg_feature = F.interpolate(patch_sseg_feature, size=(14, 14), mode='bilinear', align_corners=False) # 1 x 256 x 14 x 14
		#print('patch_sseg_feature.shape = {}'.format(patch_sseg_feature.shape))
		batch_sseg_feature[0] = patch_sseg_feature

		# rescale sseg label to 28x28
		sseg_label_patch = cv2.resize(sseg_label_proposal, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
		sseg_label_patch = sseg_label_patch.astype('int')
		#print('sseg_label_patch = {}'.format(sseg_label_patch))
		batch_sseg_label[0] = torch.tensor(sseg_label_patch)

		if self.rep_style == 'both':
			patch_feature = torch.cat((mask_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature

		#print('patch_feature.shape = {}'.format(patch_feature.shape))

		return patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal
