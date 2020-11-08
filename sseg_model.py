import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSegHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(SSegHead, self).__init__()
		self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
		self.bn5 = nn.BatchNorm2d(256)
		self.predictor = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.deconv(x)))
		x = self.predictor(x)
		return x

class DropoutHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(DropoutHead, self).__init__()
		self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
		self.bn5 = nn.BatchNorm2d(256)
		self.predictor = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = F.relu(self.bn5(self.deconv(x)))
		x = F.dropout2d(x, p=0.2, training=True)
		x = self.predictor(x)
		return x

class DuqHead(nn.Module):
	def __init__(self, num_classes=8, input_dim=512):
		super(DuqHead, self).__init__()
		self.num_classes = num_classes

		self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(256)
		self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
		
		#==========================================================================================================
		self.duq_centroid_size = 512
		self.duq_model_output_size = 256
		self.gamma = 0.999
		self.duq_length_scale = 0.1

		self.W = nn.Parameter(torch.zeros(self.duq_centroid_size, self.num_classes, self.duq_model_output_size))
		nn.init.kaiming_normal_(self.W, nonlinearity='relu')
		self.register_buffer('N', torch.ones(self.num_classes)*20)
		self.register_buffer('m', torch.normal(torch.zeros(self.duq_centroid_size, self.num_classes), 0.05))
		self.m = self.m *self.N
		self.sigma = self.duq_length_scale


	def rbf(self, z):
		z = torch.einsum('ij,mnj->imn', z, self.W)
		embeddings = self.m / self.N.unsqueeze(0)
		diff = z - embeddings.unsqueeze(0)
		diff = (diff ** 2).mean(1).div(2 * self.sigma **2).mul(-1).exp()
		return diff

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = self.deconv(x) # B x 256 x 28 x 28

		B, C, H, W = x.shape

		z = x.permute(0, 2, 3, 1)
		z = z.reshape(-1, C)

		y_pred = self.rbf(z)
		y_pred = y_pred.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)
		
		return y_pred

	def update_embeddings(self, x, y_targets):
		y_targets = y_targets.reshape(-1, 1).long().squeeze(1)
		idx_unignored = (y_targets < 255)
		y_targets = y_targets[idx_unignored]
		y_targets = F.one_hot(y_targets, self.num_classes).float()

		self.N = self.gamma * self.N + (1-self.gamma) * y_targets.sum(0)

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = self.deconv(x) # B x 256 x 28 x 28

		B, C, _, _ = x.shape

		z = x.permute(0, 2, 3, 1)
		z = z.reshape(-1, C)

		z = z[idx_unignored]

		z = torch.einsum('ij,mnj->imn', z, self.W)
		embedding_sum = torch.einsum('ijk,ik->jk', z, y_targets)

		self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum


def calc_gradient_penalty(x, y_pred):
	B, H, W, C = y_pred.shape
	y_pred = y_pred.reshape(B, -1)

	gradients = torch.autograd.grad(
		outputs=y_pred,
		inputs = x,
		grad_outputs=torch.ones_like(y_pred)/(1.0*H*W),
		create_graph=True,
	)[0]

	gradients = gradients.flatten(start_dim=1)

	grad_norm = gradients.norm(2, dim=1)

	gradient_penalty = ((grad_norm-1)**2).mean()

	return gradient_penalty