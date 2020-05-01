%%time

import os
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torch
from torch.nn import functional as F
from torch.nn.functional import conv2d
import numpy as np
from torchvision.transforms import ToTensor
from tqdm import tqdm
from os import path
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import torchvision.models as models
from torch import nn
from os import path
import pickle
from os import path
from tempfile import TemporaryFile

folder = path.expanduser('datasets/FID-300')

tracks = path.join(folder,'tracks_cropped_Subset')
refs = path.join(folder,'Subset')
PCA_refs = path.join(folder,'PCA_reference_Subset')

ref_l = [f for f in os.listdir(refs) if f.endswith('.png')]
track_l = [f for f in os.listdir(tracks) if f.endswith('.jpg')]
PCA_ref_l = [f for f in os.listdir(PCA_refs) if f.endswith('.png')]

device = torch.device('cuda:0')

googlenet = models.googlenet(pretrained=True)
model = nn.Sequential(*list(googlenet.children())[0:4])
model.to(device)
model.eval()
    
trans = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

"""    
Normalized Cross-Correlation for pattern matching.
pytorch implementation
roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

import logging
import torch
from torch.nn import functional as F


ncc_logger = logging.getLogger(__name__)


def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local means computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    
#   padding = (0,0)
    padding = tuple(side // 2 for side in patch_size)
    #padding = tuple(min(10, side // 2) for side in patch_size)
#     padding = tuple(0 for side in template_shape)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)
    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False)+10e-10)

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    """
    Computes the [Zero-Normalized Cross-Correlation][1] between an image and a template.
    Example:
        >>> lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        >>> lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1).cuda()
        >>> patch_center = 275, 275
        >>> y1, y2 = patch_center[0] - 25, patch_center[0] + 25
        >>> x1, x2 = patch_center[1] - 25, patch_center[1] + 25
        >>> lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
        >>> ncc = NCC(lena_patch)
        >>> ncc_response = ncc(lena_tensor[None, ...])
        >>> ncc_response.max()
        tensor(1.0000, device='cuda:0')
        >>> np.unravel_index(ncc_response.argmax(), lena_tensor.shape)
        (0, 275, 275)
    [1]: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        
        #self.padding = (0,0)
        self.padding = tuple(side // 2 for side in template_shape)
        #print(self.padding)
        #self.padding = tuple(min(10, side // 2) for side in template_shape)
        #print("self.padding:", self.padding)
#         self.padding = tuple(0 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        
        std = patch_std(image, self.normalized_template.shape[1:])
    
        result.div_(std+10e-10)
        if not self.keep_channels:
            result = result.mean(dim=1)
        
        # remove nan values due to sqrt of negative value
        result[result != result] = 0
        result[result != result] = result.min()

        return result

        
score_mat = np.zeros((len(np.sort(track_l)), (len(np.sort(ref_l)))), dtype='float64')

for x, t in enumerate(tqdm(np.sort(track_l))):
    for y, r in enumerate(np.sort(ref_l)):
        image = Image.open(path.join(refs, r))
        template = Image.open(path.join(tracks,t))
        
        template_t = model(trans(template).unsqueeze(0).to(device))[0]
        image_t = model(trans(image).unsqueeze(0).to(device))

        template_t2 = template_t[:, 3:template_t.shape[1]-3, 3:template_t.shape[2]-3].to(device)
        image_t2 = image_t[:,:,3:image_t.shape[2]-3, 3:image_t.shape[3]-3].to(device)

        ncc = NCC(template_t2)
        ncc_response = ncc(image_t2)
        score_mat[x][y] = np.amax(ncc_response.cpu().data.numpy())
    
np.save('scores.npy', score_mat)
