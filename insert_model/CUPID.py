import torch.nn as nn
import torch
from torch import Tensor


class ResidualConvBlock(nn.Module):
	def __init__(self, channels: int) -> None:
		super(ResidualConvBlock, self).__init__()
		self.rcb = nn.Sequential(
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
		)

	def forward(self, x: Tensor) -> Tensor:
		out = self.rcb(x)
		out = torch.add(out, x)

		return out



class CUPID(nn.Module):
	def __init__(self, in_channels=512, out_channels=512, output_size=2, trunck_num = 16,linearnum = 256):
		super(CUPID, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(in_channels, 64,kernel_size=9, stride=1, padding=4),
			nn.PReLU(),
		)
		# Features trunk blocks.
		trunk = []
		for _ in range(trunck_num):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
		)

		# Reconstruct Branch.
		self.conv_block3_mu = nn.Sequential(nn.Conv2d(64, out_channels=64,kernel_size=9, stride=1, padding=4),
			nn.ReLU(),
			nn.Conv2d(64, out_channels=out_channels,kernel_size=9, stride=1, padding=4),
			nn.ReLU(),
		)

		# Uncertainty Branch.
		self.conv_block3_sigma = nn.Sequential(
			nn.Conv2d(64, 64,kernel_size=9, stride=1, padding=4),
			nn.PReLU(),
			nn.Conv2d(64, 64,kernel_size=9, stride=1, padding=4),
			nn.PReLU(),
			nn.Conv2d(64, 1,kernel_size=9, stride=1, padding=4),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(linearnum, 1024),
			nn.ReLU(),
			nn.Linear(1024, output_size),
		)

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor):
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		out_mu = self.conv_block3_mu(out)
		out_sigma = self.conv_block3_sigma(out)
		return out_mu, [out_sigma]

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

