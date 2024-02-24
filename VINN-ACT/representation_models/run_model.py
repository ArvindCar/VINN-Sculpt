import torch
from torch import nn
from torchvision import models
import sys
pointbert_path = '/'.join('/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/VINN-ACT'.split('/')[:-3]) + '/point_cloud_embedding'
sys.path.append(pointbert_path)
# from PointBERTwEncoder import PointBERTWithProjection

class Identity(nn.Module):
    '''
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder():

    def __init__(self, params):

        self.params = params

        if(params['model'] == 'VICReg'):

            if(params['architecture'] == 'ResNet'):
                self.model = models.resnet50(pretrained=False)
            if(params['architecture'] == 'AlexNet'):
                self.model = models.alexnet(pretrained=False)

            if(params['layer'] == 'avgpool'):
                if(params['architecture'] == 'ResNet'):
                    self.model.fc = Identity()
                if(params['architecture'] == 'AlexNet'):
                    self.model.classifier = Identity()

            encoder_state_dict = torch.load(params['root_dir'] + params['representation_model_path'], map_location=torch.device('cpu'))
            self.model.load_state_dict(encoder_state_dict['model_state_dict'])


        if(params['model'] == 'BYOL'):

            if(params['architecture'] == 'ResNet'):
                self.model = models.resnet50(pretrained=False)
            if(params['architecture'] == 'AlexNet'):
                self.model = models.alexnet(pretrained=False)

            encoder_state_dict = torch.load(params['root_dir'] + params['representation_model_path'], map_location=torch.device('cpu'))
            self.model.load_state_dict(encoder_state_dict['model_state_dict'])
            if(params['layer'] == 'avgpool'):
                if(params['architecture'] == 'ResNet'):
                    self.model.fc = Identity()
                if(params['architecture'] == 'AlexNet'):
                    self.model.classifier = Identity()

        if(params['model'] == 'PointBERT'):
            pointbert_path = '/'.join('/home/arvind/CMU/MAIL/VINN/VINN-Main/VINN-Sculpt/VINN-ACT'.split('/')[:-3]) + '/point_cloud_embedding'
            self.model = PointBERTWithProjection(pointbert_path=pointbert_path)
            encoder_state_dict = torch.load(params['root_dir'] + params['representation_model_path'], map_location=torch.device('cpu'))
            self.model.load_state_dict(encoder_state_dict['model_state_dict'])
            # self.model.fc = Identity()

        if(params['model'] == 'ImageNet'):
            if(params['architecture'] == 'ResNet'):
                self.model = models.resnet50(pretrained=True)
                self.model.fc = Identity()
            if(params['architecture'] == 'AlexNet'):
                self.model = models.alexnet(pretrained=False)
                self.model.classifier = Identity()

        if(params['model'] == 'SIMClr'):
            # TODO
            pass
        if(params['model'] == 'VICReg'):
            # TODO
            pass

        if(params['eval'] == 1):
            self.model.eval()

    def encode(self, x):
            if x.numel() == 2048*3:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                x=x.float().to(device)
                # print(x.reshape(1,self.params['img_size'],3).contiguous().dtype)
                return(self.model(x.reshape(1,self.params['img_size'],3).contiguous()))
            return(self.model(x.reshape(1,3,self.params['img_size'],self.params['img_size'])))
    
            
