import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2] ## whats this line?
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0) # is this layer to CNN?

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, ker, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1) # ([16, 512, 49])
        patch_feats = patch_feats.permute(0, 2, 1)
        return patch_feats, avg_feats

        # visual extractor to extract patch features with the dimension of each feature set to 2,048.
