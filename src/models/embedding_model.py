import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import swin_v2_b, swin_v2_s, swin_v2_t

from . import fast_MPN_COV_wrapper


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, middle, nlayers = 2, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        # Modify the in_features and out_features of Linear layers
        # self.linear1 = nn.Linear(d_model, middle)
        # self.norm1 = nn.BatchNorm1d(middle, affine=True)
        # self.act = nn.ReLU(inplace=True)


        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)

        # self.linear2 = nn.Linear(middle, d_model)
        # self.act1 = nn.Tanh()

    def forward(self, x):
        # x = self.norm1(self.linear1(x))
        # x = self.act(x)
        x = self.transformer_encoder(x)

        # x = self.linear2(x)
        # x = self.act1(x)
        return x

class EmbeddingModel(nn.Module):
    def __init__(self, network='resnet18', pooling='CBP', dropout_p=0.5, cont_dims=2048, pretrained=True, middle=1000, skip_emb=False):
        super(EmbeddingModel, self).__init__()

        self.base_model = fast_MPN_COV_wrapper.get_model(arch=network, repr_agg=pooling, num_classes=cont_dims, pretrained=pretrained)
        
        # self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        

        self.emb = CustomTransformerEncoderLayer(d_model=cont_dims, nhead=4, middle=middle, dropout=0.1)

        self.out_features = cont_dims
        
        self.dropout = nn.Dropout(p=dropout_p)
        # if skip_emb:  
        #     self.emb = None
        # else:
        #     self.emb = nn.Sequential(
        #         nn.Linear(cont_dims, middle),
        #         nn.BatchNorm1d(middle, affine=True),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(middle, cont_dims),
        #         nn.Tanh()
        #         )

    def forward(self, x):

        x = self.base_model(x)
        x = self.emb(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)

if __name__ == '__main__':
    model = EmbeddingModel(cont_dims=2048)

    inp = torch.randn(4, 3, 250, 250)

    outputs = model(inp)

    print(outputs.shape)

    #print(list(model.base_model.children()))
