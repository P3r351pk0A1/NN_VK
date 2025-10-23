
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class SSDLiteHead(nn.Module):
	def __init__(self, in_channels, num_anchors, num_classes):
		super().__init__()
		self.in_channels = in_channels
		self.num_anchors = num_anchors
		self.num_classes = num_classes
		# SSDLite head: depthwise separable conv
		self.loc = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, num_anchors * 4, 1)
		)
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, num_anchors * num_classes, 1)
		)

	def forward(self, x):
		loc = self.loc(x)
		cls = self.cls(x)
		# (B, A*4, H, W) -> (B, H*W*A, 4)
		B = x.shape[0]
		loc = loc.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
		# classification: (B, A*C, H, W) -> (B, H*W*A, C)
		cls = cls.permute(0, 2, 3, 1).contiguous()
		cls = cls.view(B, -1, self.num_classes)
		return loc, cls

class SSDLiteMobileNetV2(nn.Module):
	def __init__(self, num_classes=2, pretrained_backbone=True, num_anchors=6):
		super().__init__()
		self.num_classes = num_classes
		self.num_anchors = num_anchors
		# Backbone
		backbone = mobilenet_v2(weights='IMAGENET1K_V1' if pretrained_backbone else None)
		self.backbone = backbone.features
		# Extra feature maps built on top of backbone's last feature
		self.extra = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(1280, 512, 1), nn.ReLU(inplace=True),
				nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
			nn.Sequential(
				nn.Conv2d(512, 256, 1), nn.ReLU(inplace=True),
				nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True)),
		])

		# create heads later based on actual channel sizes of feature maps
		# run a dummy forward on CPU to infer feature channels
		self.heads = nn.ModuleList()
		try:
			self.eval()
			with torch.no_grad():
				dummy = torch.zeros(1, 3, 320, 320)
				feats = []
				x = dummy
				for i, layer in enumerate(self.backbone):
					x = layer(x)
					if i == 6:
						feats.append(x)
					if i == 13:
						feats.append(x)
				y = x
				for extra in self.extra:
					y = extra(y)
					feats.append(y)
				# create heads matching channels
				for f in feats:
					ch = f.shape[1]
					self.heads.append(SSDLiteHead(ch, num_anchors, num_classes))
		except Exception:
			# fallback to defaults if something goes wrong
			self.heads = nn.ModuleList([
				SSDLiteHead(320, num_anchors, num_classes),
				SSDLiteHead(96, num_anchors, num_classes),
				SSDLiteHead(512, num_anchors, num_classes),
				SSDLiteHead(256, num_anchors, num_classes),
			])

	def get_feature_map_shapes(self, image_size=(320, 320)):
		"""Run a dummy forward to get feature map shapes (H, W) for each feature used by heads.

		Returns list of (H, W) tuples corresponding to feats in forward.
		"""
		device = next(self.parameters()).device
		dummy = torch.zeros(1, 3, image_size[0], image_size[1]).to(device)
		feats = []
		x = dummy
		for i, layer in enumerate(self.backbone):
			x = layer(x)
			if i == 6:
				feats.append(x)
			if i == 13:
				feats.append(x)
		y = x
		for extra in self.extra:
			y = extra(y)
			feats.append(y)
		shapes = [(f.shape[2], f.shape[3]) for f in feats]
		return shapes

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

	def forward(self, x):
		feats = []
		# Backbone feature maps
		for i, layer in enumerate(self.backbone):
			x = layer(x)
			if i == 6:  # conv4_3
				feats.append(x)
			if i == 13: # conv7
				feats.append(x)
		# Extra feature maps
		y = x
		for extra in self.extra:
			y = extra(y)
			feats.append(y)
		locs, clss = [], []
		for f, head in zip(feats, self.heads):
			loc, cls = head(f)
			locs.append(loc)
			clss.append(cls)
		locs = torch.cat(locs, dim=1)
		clss = torch.cat(clss, dim=1)
		return locs, clss

def ssdlite_mobilenetv2(num_classes=2, pretrained_backbone=True, num_anchors=6):
	model = SSDLiteMobileNetV2(num_classes=num_classes, pretrained_backbone=pretrained_backbone, num_anchors=num_anchors)
	model.freeze_bn()
	return model
