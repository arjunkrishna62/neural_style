import torch
import torch.nn as nn
from torchvision import models # type: ignore
from collections import namedtuple

#from typing import Union, List, Dict, Any, cast


"""
Useful sources:
    * https://arxiv.org/pdf/1409.1556 (original VGG paper)
    * https://arxiv.org/abs/1508.06576 (original nst paper)
    * https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
    * https://github.com/gordicaleksa/pytorch-neural-style-transfer/blob/master/models/definitions/vgg_nets.py
    * https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
"""

#torch.hub.get_dir() # path were the model is saved

class Vgg16(nn.Module):
    
    def __init__(self):
        super().__init__()
        # torch_vision vgg net has 3 sub-modules (features, avg pooling and
        # classifier), we are only interested in the conv layers, so:
        vgg_pretrained_features = models.vgg16(pretrained=True, 
                                               progress=True).features
        # Layers used for content and style loss:
        self.slice1 = vgg_pretrained_features[:4]
        self.slice2 = vgg_pretrained_features[4:9]
        self.slice3 = vgg_pretrained_features[9:16]
        self.slice4 = vgg_pretrained_features[16:23]
        
        # their names: layeri_j (resolution i, layer j):
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        # we optimize the noisy input image, not the parameters, so:
        for param in self.parameters():
            param.requires_grad=False
            
    def forward(self, x):
        x_relu1_2 = self.slice1(x)
        x_relu2_2 = self.slice2(x_relu1_2)
        x_relu3_3 = self.slice3(x_relu2_2)
        x_relu4_3 = self.slice4(x_relu3_3)
        # collect all outputs:
        vggOutputs = namedtuple('VggOutputs', self.layer_names)
        out = vggOutputs(x_relu1_2, x_relu2_2, x_relu3_3, x_relu4_3)        
        return out


class Vgg19(torch.nn.Module):

    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True, 
                                               progress=True).features        
        # their names: layeri_j (resolution i, layer j):
        self.layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
        self.content_feature_maps_index = 4  # conv4_2        
        # all layers used for style representation except conv4_2
        self.style_feature_maps_indices = list(range(len(self.layer_names)))
        self.style_feature_maps_indices.remove(4)  # conv4_2        
        # Layers used for content and style loss:
        self.slice1 = vgg_pretrained_features[:2]
        self.slice2 = vgg_pretrained_features[2:7]
        self.slice3 = vgg_pretrained_features[7:12]
        self.slice4 = vgg_pretrained_features[12:21]
        self.slice5 = vgg_pretrained_features[21:22]
        self.slice6 = vgg_pretrained_features[22:30]
        # we optimize the noisy input image, not the parameters, so:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_relu1_1 = self.slice1(x)
        x_relu2_1 = self.slice2(x_relu1_1)
        x_relu3_1 = self.slice3(x_relu2_1)
        x_relu4_1 = self.slice4(x_relu3_1)
        x_relu4_2 = self.slice5(x_relu4_1)
        x_relu5_1 = self.slice6(x_relu4_2)
        # collect all outputs:
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(x_relu1_1, x_relu2_1, x_relu3_1, x_relu4_1,
                          x_relu4_2, x_relu5_1)
        return out



class Vgg19_raw(nn.Module):    
    
    # convnet configuration type E (see VGG paper, table 1):
    cfg = 2*[64]+['M']+2*[128]+['M']+4*[256]+['M']+4*[512]+['M']+4*[512]+['M']
    
    def __init__(self):
        super().__init__()
        # Convolutional features:
        self.features = Vgg19_raw.vgg_conv_layers(Vgg19_raw.cfg)
        # for classification:
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,1000),
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1) # (x,1) in order to respect batches
        x = self.classifier(x)
        return 
        

    @staticmethod
    def vgg_conv_layers(cfg):
        current_channel_inputs=3 # RGB image
        layers = []
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=current_channel_inputs,
                                    out_channels=l,
                                    kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                current_channel_inputs = l
        return nn.Sequential(*layers)
    
    
if __name__ == "__main__":
    model = Vgg16()
    x = torch.rand(1,3,224,224) # random image
    outs = model(x)
    print('layer\tshape')
    for out, name in zip(outs, model.layer_names):
        print(f'{name} {out.shape}')
        