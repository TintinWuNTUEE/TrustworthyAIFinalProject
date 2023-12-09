import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self,model):
        super(FeatureExtractor, self).__init__()
        self.net = model
        for p in self.net.parameters():
            p.requires_grad = True
        # last conv of resnet
        self.features = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x
