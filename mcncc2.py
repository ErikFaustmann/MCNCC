import os
from os import path
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import pickle
import numpy as np
from tempfile import TemporaryFile
from tqdm import tqdm
import argparse
import logging
import time
import pandas as pd

import torch
from torch.nn import functional as F
from torch.nn.functional import conv2d
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
from torchvision import transforms
#from torchvision.transforms import functional as F
import torchvision.models as models
from torch import nn

folder = '/content/drive/My Drive/MCNCC/datasets/FID-300'

parser = argparse.ArgumentParser(description='take some indiviudal folders from user')
parser.add_argument('-t', '--tracks', type = str, default='tracks_cropped_Subset', help='define track folder')
parser.add_argument('-rf', '--refs', type=str, default='Subset', help='define reference folder')
parser.add_argument('-str', '--stride', type = int, default='1', help='stride for convolutions')
parser.add_argument('-avgp', '--avgpool_bool', default='False', help='activate average pooling for features')
parser.add_argument('-r', '--rot', default=False, action='store_true', help='add rotation')
parser.add_argument('-ris', '--start', type=int, default=-10, help='rotation interval start')
parser.add_argument('-rie', '--end', type=int, default=11, help='rotation interval end')
parser.add_argument('-sf', '--scorefile', type=str, default='scores.npy', help='scorefilename')
parser.add_argument('-cmc', '--cmc', default=False, action='store_true', help='calculate cmc')
parser.add_argument('-cmcf', '--cmc_file', type=str, default='cmc_file',help='cmc filename')
parser.add_argument('-lbltable', '--label_file', type=str, default='Subsetlabels.csv',help='name of the csv. file')

args = parser.parse_args()

tracks = path.join(folder, args.tracks)
refs = path.join(folder, args.refs)

ref_l = [f for f in os.listdir(refs) if f.endswith('.png')]
track_l = [f for f in os.listdir(tracks) if f.endswith('.jpg')]

if __name__ == "__main__":
    print("refs:", args.refs)
    print("tracks:", args.tracks)
    print("stride:", args.stride)
    print("rot:", args.rot)
    print("start:", args.start)
    print("end:", args.end)
    print("cmc:", args.cmc)
    print("scorefile:", args.scorefile)
    print("cmc_file:", args.cmc_file)
    print("label_file:", args.label_file)
    print("average_pooling:", args.avgpool_bool)


device = torch.device('cuda:0')

googlenet = models.googlenet(pretrained=True)
model = nn.Sequential(*list(googlenet.children())[0:4])

if args.avgpool_bool == True:
    model = nn.Sequential(model, nn.AvgPool2d(2, stride=1))
    
model.to(device)
model.eval()
    
trans = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

ncc_logger = logging.getLogger(__name__)


def compute_cmc(score_mat):
    
    test_labels = pd.read_csv(path.join(folder, args.label_file), header=None)
    test_labels = test_labels[:][1].values.tolist()
    
    true_mat = np.zeros((len(test_labels), 38))

    for i in range(len(test_labels)):
        true_mat[i, test_labels[i]-1]= 1
    
    cmc = np.zeros(score_mat.shape[1], dtype='float64' )
    mx = np.zeros(score_mat.shape[0], dtype='float64')
    true_mat_est = np.zeros(score_mat.shape)
    est_loc =np.zeros(score_mat.shape[0])
    score_mat2 = score_mat

    for i in range(score_mat.shape[1]):
#________________________________________________________________________________ 
    #mx = np.zeros(score_mat.shape[0], dtype='float64')

        for w in range(score_mat.shape[0]):
            mx[w] = max(score_mat2[w])
    #print(mx)
#________________________________________________________________________________ 
    #true_mat_est = np.zeros(score_mat.shape)

        for e in range(score_mat.shape[0]):
            true_mat_est[e]  = np.equal(score_mat2[e], mx[e])

            est_loc[e] = list(true_mat_est[e]).index(1)
        if i == 0:
            with np.printoptions(threshold=np.inf):
                #print(true_mat_est)
                print(est_loc)

        true_mat_est = true_mat_est*1
    #print(i, ":", sum(true_mat_est))
#________________________________________________________________________________     
    
        if i == 0:
            cmc[i] = np.tensordot(true_mat, true_mat_est, axes=2)/score_mat.shape[0]
            #print("correct matches:", numpy.tensordot(true_mat, true_mat_est, axes=2))
        else:
            cmc[i] = (np.tensordot(true_mat, true_mat_est, axes=2)/score_mat.shape[0])+ cmc[i-1]
            #print("correct matches:", numpy.tensordot(true_mat, true_mat_est, axes=2))
#________________________________________________________________________________ 
        for g in range(score_mat.shape[0]):
            score_mat2[g][int(est_loc[g])] = -100000
            #print(est_loc[g])
            #print(score_mat2[g])
            
    np.save("/content/drive/My Drive/MCNCC/"+args.cmc_file, cmc)

    return cmc

def patch_mean(images, patch_shape):

    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)


    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).bool()
    weights[~channel_selector] = 0

    result = conv(images, weights, stride = args.stride, padding=padding, bias=None)
    return result


def patch_std(image, patch_shape):
    
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):

    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False)+10e-10)

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):

    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).bool()
        self.normalized_template[~channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, stride = args.stride, padding=self.padding, bias=None)
        
        std = patch_std(image, self.normalized_template.shape[1:])
    
        result.div_(std+10e-10)
        if not self.keep_channels:
            result = result.mean(dim=1)
        
        # remove nan values due to sqrt of negative value
        result[result != result] = 0
        result[result != result] = result.min()

        return result

score_mat = np.zeros((len(np.sort(track_l)), (len(np.sort(ref_l)))), dtype='float64')

calc_time = time.time()

if args.rot == False:

    for x, t in enumerate(tqdm(np.sort(track_l))):

        template = Image.open(path.join(tracks,t))
        template_t = model(trans(template).unsqueeze(0).to(device))[0]
        template_t2 = template_t[:, 3:template_t.shape[1]-3, 3:template_t.shape[2]-3]
        ncc = NCC(template_t2)
        
        for y, r in enumerate(np.sort(ref_l)):
            image = Image.open(path.join(refs, r))
            image_t = model(trans(image).unsqueeze(0).to(device))
            image_t2 = image_t[:,:,3:image_t.shape[2]-3, 3:image_t.shape[3]-3].to(device)

            ncc_response = ncc(image_t2)
            score_mat[x][y] = np.amax(ncc_response.cpu().data.numpy())

else:
    
    for x, t in enumerate(tqdm(np.sort(track_l))):
        
        template = Image.open(path.join(tracks,t))
        template_t = model(trans(template).unsqueeze(0).to(device))[0]
        template_t2 = template_t[:, 3:template_t.shape[1]-3, 3:template_t.shape[2]-3]
        ncc = NCC(template_t2)

        for y, r in enumerate(np.sort(ref_l)):
            
            img = Image.open(path.join(refs, r))
            for i in range(args.start, args.end):
                if i == args.start:
                    img2 = img.rotate(i, expand=1, fillcolor='white')
                    img2 = trans(img2).unsqueeze(0)
                    img_batch = torch.zeros(abs(args.start-args.end), 3, img2.shape[2], img2.shape[3])
                else:
                    img2 = img.rotate(i)
                    img2 = trans(img2).unsqueeze(0)
                    img_batch[i][:, :img2[0].shape[1],:img2[0].shape[2]] = img2[0]

            img_t_batch = model(img_batch.to(device))

            ncc_response = ncc(img_t_batch) 
            
            
            cc = 0

            for i in range(abs(args.start-args.end)-1):
                if cc < torch.max(ncc_response[i]).item():
                    cc = torch.max(ncc_response[i]).item()
                    
            score_mat[x][y] = cc
    
elapsed = time.time() - calc_time
print("elapsed time:", elapsed)
np.save("/content/drive/My Drive/MCNCC/"+args.scorefile, score_mat)
print("score_mat saved")

if args.cmc == True:
    
    cmc_score = compute_cmc(score_mat)
    
    f, ax = plt.subplots(1)
    plt.plot(cmc_score)
    plt.xlabel('references')
    plt.ylabel('Correleation')
    ax.set_ylim(bottom=0)
    plt.grid(True)
    f.savefig('/content/drive/My Drive/MCNCC/cmc_score_diagram')
