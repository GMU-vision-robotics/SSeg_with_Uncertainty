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
from torchvision.ops import roi_align

device = torch.device('cuda')

class CityscapesProposalsDataset(data.Dataset):
	def __init__(self, dataset_dir, split='train', batch_size=3, rep_style='both'):

		self.dataset_dir = dataset_dir
		self.split = split
		self.mode = split
		self.batch_size = batch_size
		self.rep_style = rep_style

		self.img_list = np.load('{}/{}_img_list.npy'.format(self.dataset_dir, self.mode), allow_pickle=True).tolist()

		self.void_classes = [0, 1, 2, 3, 4, 5, 10, 14, 15, 16, -1]
		self.valid_classes = [7, 11, 17, 21, 23, 24, 26, 31]
		self.class_names = ['unlabelled', 'road', 'building', \
							'pole', 'vegetation', 'sky', 'person', 'car', 'train', ]

		self.ignore_index = 255
		self.NUM_CLASSES = len(self.valid_classes)
		self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

		print("Found {} {} images".format(len(self.img_list), self.split))

		# proposal, mask feature and sseg feature folder
		self.proposal_folder = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/generated_proposals/cityscapes_{}'.format(self.mode)
		self.mask_ft_folder  = '/scratch/yli44/detectron2/my_projects/Bayesian_MaskRCNN/proposal_mask_features/cityscapes_{}'.format(self.mode)
		self.sseg_ft_folder  = '/projects/kosecka/yimeng/Datasets/Cityscapes/deeplab_ft_8_classes/{}'.format(self.mode)

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, i):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['rgb_path'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['semSeg_path'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		#print('mask_feature.shape = {}'.format(mask_feature.shape))
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		_, H, W = sseg_feature.shape
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256

		index = np.random.choice(100, self.batch_size, replace=False)
		#index = np.array([0,1,2])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		#print('proposals.shape = {}'.format(proposals.shape))
		#print('mask_feature.shape = {}'.format(mask_feature.shape))

		#batch_sseg_feature = torch.zeros((self.batch_size, 256, 14, 14))
		batch_sseg_label = torch.zeros((self.batch_size, 28, 28))
		batch_prop_boxes = torch.zeros((self.batch_size, 4)) 

		for j in range(self.batch_size):
			x1, y1, x2, y2 = proposals[j]
			prop_x1 = int(max(round(x1), 0))
			prop_y1 = int(max(round(y1), 0))
			prop_x2 = int(min(round(x2), 2048-1))
			prop_y2 = int(min(round(y2), 1024-1))

			img_patch = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
			sseg_label_patch = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

			batch_prop_boxes[j, 0] = prop_x1
			batch_prop_boxes[j, 1] = prop_y1
			batch_prop_boxes[j, 2] = prop_x2
			batch_prop_boxes[j, 3] = prop_y2
			
			'''
			# visualize for test
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
			ax[0].imshow(img_patch)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb image")
			ax[1].imshow(sseg_label_patch, vmin=0.0, vmax=8.0)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg label")
			plt.show()
			'''

			# rescale sseg label to 28x28
			sseg_label_patch = cv2.resize(sseg_label_patch, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
			#print('sseg_label_patch.shape = {}'.format(sseg_label_patch.shape))
			batch_sseg_label[j] = torch.tensor(sseg_label_patch)

		batch_prop_boxes = batch_prop_boxes.to(device)
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0)
		#print('batch_sseg_feature.shape = {}'.format(batch_sseg_feature.shape))

		if self.rep_style == 'both':
			patch_feature = torch.cat((mask_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature 
		#print('patch_feature.shape = {}'.format(patch_feature.shape))

		return patch_feature, batch_sseg_label
		#assert 1==2

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

	def get_proposal(self, i, j=0):
		img_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['rgb_path'])
		lbl_path = '{}/{}'.format(self.dataset_dir, self.img_list[i]['semSeg_path'])

		rgb_img = np.array(Image.open(img_path).convert('RGB'))
		sseg_label = np.array(Image.open(lbl_path), dtype=np.uint8)
		sseg_label = self.encode_segmap(sseg_label) # 1024 x 2048
		#print('sseg_label.shape = {}'.format(sseg_label.shape))
		
		# read proposals
		proposals = np.load('{}/{}_proposal.npy'.format(self.proposal_folder, i), allow_pickle=True)
		# read mask features
		mask_feature = np.load('{}/{}_proposal_mask_features.npy'.format(self.mask_ft_folder, i), allow_pickle=True)
		# read sseg features
		sseg_feature = np.load('{}/{}_deeplab_ft.npy'.format(self.sseg_ft_folder, i), allow_pickle=True) # 256 x 128 x 256
		_, H, W = sseg_feature.shape
		#print('sseg_feature.shape = {}'.format(sseg_feature.shape))

		sseg_feature = torch.tensor(sseg_feature).unsqueeze(0).to(device) # 1 x 256 x 128 x 256
		
		index = np.array([j])
		proposals = proposals[index] # B x 4
		mask_feature = torch.tensor(mask_feature[index]).to(device) # B x 256 x 14 x 14

		batch_sseg_label = torch.zeros((1, 28, 28))
		batch_prop_boxes = torch.zeros((1, 4))
		
		x1, y1, x2, y2 = proposals[0]
		prop_x1 = int(max(round(x1), 0))
		prop_y1 = int(max(round(y1), 0))
		prop_x2 = int(min(round(x2), 2048-1))
		prop_y2 = int(min(round(y2), 1024-1))

		img_proposal = rgb_img[prop_y1:prop_y2, prop_x1:prop_x2]
		sseg_label_proposal = sseg_label[prop_y1:prop_y2, prop_x1:prop_x2]

		batch_prop_boxes[0, 0] = prop_x1
		batch_prop_boxes[0, 1] = prop_y1
		batch_prop_boxes[0, 2] = prop_x2
		batch_prop_boxes[0, 3] = prop_y2
		
		'''
		# visualize for test
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
		ax[0].imshow(img_proposal)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb proposal")
		ax[1].imshow(sseg_label_proposal, vmin=0.0, vmax=8.0)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg label proposal")
		plt.show()
		'''

		# rescale sseg label to 28x28
		sseg_label_patch = cv2.resize(sseg_label_proposal, (28, 28), interpolation=cv2.INTER_NEAREST) # 28 x 28
		sseg_label_patch = sseg_label_patch.astype('int')
		#print('sseg_label_patch = {}'.format(sseg_label_patch))
		batch_sseg_label[0] = torch.tensor(sseg_label_patch)

		batch_prop_boxes = batch_prop_boxes.to(device)
		batch_sseg_feature = roi_align(sseg_feature, [batch_prop_boxes], output_size=(14, 14), spatial_scale=1/8.0)

		if self.rep_style == 'both':
			patch_feature = torch.cat((mask_feature, batch_sseg_feature), dim=1) # B x 512 x 14 x 14
		elif self.rep_style == 'ObjDet':
			patch_feature = mask_feature
		elif self.rep_style == 'SSeg':
			patch_feature = batch_sseg_feature

		return patch_feature, batch_sseg_label, img_proposal, sseg_label_proposal

'''
cityscapes_train = CityscapesProposalsDataset('/projects/kosecka/yimeng/Datasets/Cityscapes', 'train')
a = cityscapes_train[1]
#b = cityscapes_train.get_proposal(0, 2)
'''