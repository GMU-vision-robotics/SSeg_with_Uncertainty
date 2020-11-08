import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sseg_model import DropoutHead
from dataloaders.cityscapes_proposals import CityscapesProposalsDataset
from dataloaders.lostAndFound_proposals import LostAndFoundProposalsDataset
import torch.nn.functional as F
from utils import apply_color_map
from scipy.stats import entropy
from scipy.special import softmax

style = 'dropout'
dataset = 'lostAndFound' #'cityscapes'
rep_style = 'ObjDet'

saved_folder = 'visualization/obj_sseg_{}/{}/{}'.format(style, rep_style, dataset)
trained_model_dir = 'trained_model/{}'.format(style)
num_forward_pass = 10

if dataset == 'cityscapes':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Cityscapes'
	ds_val = CityscapesProposalsDataset(dataset_folder, 'val', rep_style=rep_style)
elif dataset == 'lostAndFound':
	dataset_folder = '/projects/kosecka/yimeng/Datasets/Lost_and_Found'
	ds_val = LostAndFoundProposalsDataset(dataset_folder, rep_style=rep_style)
num_classes = ds_val.NUM_CLASSES

if rep_style == 'both':
    input_dim = 512
else:
    input_dim = 256

device = torch.device('cuda')

classifier = DropoutHead(num_classes, input_dim).to(device)
classifier.load_state_dict(torch.load('{}/{}_classifier.pth'.format(trained_model_dir, style)))

with torch.no_grad():
	for i in range(len(ds_val)):
		if dataset == 'cityscapes':
			num_proposals = 10
		elif dataset == 'lostAndFound':
			num_proposals = ds_val.get_num_proposal(i)

		for j in range(num_proposals):
			print('i = {}, j = {}'.format(i, j))
			patch_feature, _, img_proposal, sseg_label_proposal = ds_val.get_proposal(i, j)
			H, W = sseg_label_proposal.shape

			pass_logits = torch.zeros((num_forward_pass, num_classes, H, W))

			patch_feature = patch_feature.to(device)
			for p in range(num_forward_pass):
				logits = classifier(patch_feature)
				logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
				pass_logits[p] = logits[0]
			
			logits = torch.mean(pass_logits, dim=0)
			sseg_pred = torch.argmax(logits, dim=0)

			logits = logits.cpu().numpy()
			sseg_pred = sseg_pred.cpu().numpy()

			uncertainty = entropy(softmax(logits, axis=0), axis=0, base=2)

			if dataset == 'cityscapes':
				color_sseg_label_proposal = apply_color_map(sseg_label_proposal)
			else:
				color_sseg_label_proposal = sseg_label_proposal
			color_sseg_pred = apply_color_map(sseg_pred)
			#assert 1==2

			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18,10))
			ax[0][0].imshow(img_proposal)
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title("rgb proposal")
			ax[0][1].imshow(color_sseg_label_proposal)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title("sseg_label_proposal")
			ax[1][0].imshow(color_sseg_pred)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title("sseg pred")
			ax[1][1].imshow(uncertainty, vmin=0.0, vmax=3.0)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title("uncertainty")

			fig.tight_layout()
			fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
			plt.close()
		

			#assert 1==2

