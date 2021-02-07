import torch.nn as nn
import torch
import torch.nn.functional as F


class EffNetb0(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(EffNetb0, self).__init__()
        self.model_name = 'effnet'
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class
        # Create model
        torch.hub.list('rwightman/gen-efficientnet-pytorch')
        self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=False)
        
        self.effnet.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Modify last linear layer
        num_ftrs = self.effnet.classifier.in_features
        self.effnet.classifier = nn.Linear(num_ftrs, 2+pitch_class+pitch_octave+2)

        
    def forward(self, x):
        out = self.effnet(x)
        # print(out.shape)
        # [batch, output_size]

        onset_logits = out[:, 0]
        offset_logits = out[:, 1]

        pitch_out = out[:, 2:]
        
        pitch_octave_logits = pitch_out[:, 0:self.pitch_octave+1]
        pitch_class_logits = pitch_out[:, self.pitch_octave+1:]

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits


if __name__ == '__main__':
    from torchsummary import summary
    model = EffNetb0().cuda()
    summary(model, input_size=(1, 11, 168))
