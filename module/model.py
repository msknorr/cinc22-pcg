import timm
import torch
from torch import nn as nn
from torch.nn import functional as F


class PretrainModel(nn.Module):
    """
        Pretraining using this model in the Jupyter Notebook on Cinc2016 data
    """

    def __init__(self):
        super().__init__()

        model = timm.create_model('efficientnet_b4', in_chans=3, pretrained=True)
        self.last_layer_shape = model.classifier.in_features
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.ln1 = nn.Linear(self.last_layer_shape, 2)

    def forward(self, x, x2):
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        x = self.ln1(x)
        return x


class Backbone(nn.Module):

    def __init__(self, backbone_last_layer_dim):
        super().__init__()

        use_pretraining = False
        if use_pretraining:
            print("Loading Cinc16 pretrained model")
            pretrainedmodel = PretrainModel()
            cp = torch.load("./transfer_learning/best-checkpoint-fold0_0-020epoch.bin")
            pretrainedmodel.load_state_dict(cp["model_state_dict"])
            self.last_layer_shape = pretrainedmodel.last_layer_shape
            self.model = pretrainedmodel.model
        else:
            model = timm.create_model('efficientnet_b1', in_chans=3, pretrained=True)
            self.last_layer_shape = model.classifier.in_features
            self.model = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.fft = nn.Linear(self.last_layer_shape, backbone_last_layer_dim)
        self.backbone_last_layer_dim = backbone_last_layer_dim

    def forward(self, img):
        # bs, n_leads, n_crops,crop_y/nmels, crop_y/time, 3
        bs = img.shape[0]
        n_leads = img.shape[1]
        n_crops = img.shape[2]
        n_mels = img.shape[3]
        n_timepoints = img.shape[4]
        n_channels = 3

        img = img.reshape(bs * n_leads * n_crops, n_mels, n_timepoints, n_channels)

        img = img.permute(0, 3, 1, 2)
        x = self.model(img)

        x = self.fft(x)
        x = x.reshape(bs, n_leads, n_crops, self.backbone_last_layer_dim)

        x = torch.mean(x, dim=2)
        x = x.reshape(bs, -1)
        return x


class ResNet(nn.Module):

    def __init__(self, use_tab=False, backbone_last_layer_dim=25):
        super().__init__()
        self.backbone = nn.ModuleList([Backbone(backbone_last_layer_dim)])

        self.use_tab = use_tab
        if use_tab:
            self.n_tab = 13
        else:
            self.n_tab = 0

        self.intermediate_linear = nn.Linear(backbone_last_layer_dim * 4 + self.n_tab, backbone_last_layer_dim)

        self.ln0 = nn.Linear(backbone_last_layer_dim, 3)  # murmur
        self.ln1 = nn.Linear(backbone_last_layer_dim, 2)  # outcome
        self.ln2 = nn.Linear(backbone_last_layer_dim, 4)  # where hearable
        self.ln3 = nn.Linear(backbone_last_layer_dim, 5)  # where loudest

        self.ln4 = nn.Linear(backbone_last_layer_dim, 5)  # timing
        self.ln5 = nn.Linear(backbone_last_layer_dim, 5)  # shape
        self.ln6 = nn.Linear(backbone_last_layer_dim, 4)  # grading
        self.ln7 = nn.Linear(backbone_last_layer_dim, 4)  # pitch
        self.ln8 = nn.Linear(backbone_last_layer_dim, 4)  # quality

    def forward(self, img, tab):
        x = self.backbone[0](img)

        if self.use_tab:
            x = torch.cat((x, tab), dim=1)
        x = F.dropout(x, training=self.training)
        x = self.intermediate_linear(x)

        murmur = self.ln0(x)
        outcome = self.ln1(x)
        where_hearable = self.ln2(x)
        # where_loudest = self.ln3(x)
        murmur_timing = self.ln4(x)
        murmur_shape = self.ln5(x)
        murmur_grading = self.ln6(x)
        murmur_pitch = self.ln7(x)
        murmur_quality = self.ln8(x)

        return murmur, outcome, where_hearable, murmur_timing, murmur_shape, murmur_grading, murmur_pitch, murmur_quality  # , where_loudest
