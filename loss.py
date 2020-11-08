import torch
import torch.nn as nn
import torch.nn.functional as F

def BinaryCrossEntropyLoss(logit, target, num_classes=8):
	B, C, H, W = logit.shape
	logit = logit.permute(0, 2, 3, 1).reshape(-1, C)

	y_targets = target.reshape(-1, 1).long().squeeze(1)
	idx_unignored = (y_targets < 255)
	y_targets = y_targets[idx_unignored]
	y_targets = F.one_hot(y_targets, num_classes).float()

	logit = logit[idx_unignored]
	#print('logit.shape = {}, y_targets.shape = {}'.format(logit.shape, y_targets.shape))

	loss = F.binary_cross_entropy(logit, y_targets, reduction='sum').div(num_classes * logit.shape[0])
	return loss