import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNetFlow
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .unet import UNetRecurrent, UNet
from .submodules import ResidualBlock, ConvGRU, ConvLayer
from utils.color_utils import merge_channels_into_color_image



def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class ColorNet(BaseModel):
    """
    Split the input events into RGBW channels and feed them to an existing
    recurrent model with states.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = {'R': [slice(0, None, 2), slice(0, None, 2)],
                         'G': [slice(0, None, 2), slice(1, None, 2)],
                         'B': [slice(1, None, 2), slice(1, None, 2)],
                         'W': [slice(1, None, 2), slice(0, None, 2)],
                         'grayscale': [slice(None), slice(None)]}
        self.prev_states = {k: self.model.states for k in self.channels}

    def reset_states(self):
        self.model.reset_states()

    @property
    def num_encoders(self):
        return self.model.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with RGB image taking values in [0, 1], and
                 displacement within event_tensor.
        """
        height, width = event_tensor.shape[-2:]
        crop_halfres = CropParameters(int(width / 2), int(height / 2), self.model.num_encoders)
        crop_fullres = CropParameters(width, height, self.model.num_encoders)
        color_events = {}
        reconstructions_for_each_channel = {}
        for channel, s in self.channels.items():
            color_events = event_tensor[:, :, s[0], s[1]]
            if channel == 'grayscale':
                color_events = crop_fullres.pad(color_events)
            else:
                color_events = crop_halfres.pad(color_events)
            self.model.states = self.prev_states[channel]
            img = self.model(color_events)['image']
            self.prev_states[channel] = self.model.states
            if channel == 'grayscale':
                img = crop_fullres.crop(img)
            else:
                img = crop_halfres.crop(img)
            img = img[0, 0, ...].cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            reconstructions_for_each_channel[channel] = img
        image_bgr = merge_channels_into_color_image(reconstructions_for_each_channel)  # H x W x 3
        return {'image': image_bgr}

class E2VIDRecurrent(BaseModel):
    """
    E2VIDRecurrent————UNetRecurrent————BaseUNet
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict

class FlowNet(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlow(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetflow.states)

    @states.setter
    def states(self, states):
        self.unetflow.states = states

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict

