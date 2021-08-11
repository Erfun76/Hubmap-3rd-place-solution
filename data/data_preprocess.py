import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm.notebook import tqdm
import zipfile
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import gc

## HyperParameter

sz = 512   #the size of tiles
reduce = 4 #reduce the original images by 4 times
expansion = 64
MASKS = '../../kidney_challenge/input/hubmap-kidney-segmentation/train.csv'
DATA = '../../kidney_challenge/input/hubmap-kidney-segmentation/train/'
OUT_TRAIN = f'train{sz}reduce{reduce}expansion{expansion}.zip'
OUT_MASKS = f'masks{sz}reduce{reduce}expansion{expansion}.zip'

## Utils

#functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

df_masks = pd.read_csv(MASKS).set_index('id')
# data_train_available = ['1e2425f28', '2f6ecfcdf', '26dc41664']
# df_masks = df_masks[df_masks.index.isin(data_train_available)]
# df_masks

## Data Loader
# one of the new images cannot be loaded into 16GB RAM
# use rasterio to load image part by part
# using a dataset similar to my submission kernel

s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels

identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, sz_reduction=reduce, expansion=expansion, encs=None):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.sz_reduction = sz_reduction
        self.sz = sz_reduction * sz
        self.expansion = sz_reduction * expansion
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz
        self.mask = enc2mask(encs, (self.shape[1], self.shape[0])) if encs is not None else None

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz - self.expansion // 2, -self.pad1 // 2 + n1 * self.sz - self.expansion // 2
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz + self.expansion, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz + self.expansion, self.shape[1])
        img = np.zeros((self.sz + self.expansion, self.sz + self.expansion, 3), np.uint8)
        mask = np.zeros((self.sz + self.expansion, self.sz + self.expansion), np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                                           window=Window.from_slices(
                                                                                               (p00, p01), (p10, p11))),
                                                                            0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = \
                    layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))
        if self.mask is not None: mask[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = self.mask[p00:p01, p10:p11]
        if self.sz_reduction != 1:
            img = cv2.resize(img, (
            (self.sz + self.expansion) // self.sz_reduction, (self.sz + self.expansion) // self.sz_reduction),
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (
            (self.sz + self.expansion) // self.sz_reduction, (self.sz + self.expansion) // self.sz_reduction),
                              interpolation=cv2.INTER_NEAREST)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return img, mask, (-1 if (s > s_th).sum() <= p_th or img.sum() <= p_th else idx)


## data prepartion

x_tot, x2_tot = [], []
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, \
        zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
        # image+mask dataset
        ds = HuBMAPDataset(index, encs=encs)
        for i in range(len(ds)):
            im, m, idx = ds[i]
            if idx < 0: continue

            x_tot.append((im / 255.0).reshape(-1, 3).mean(0))
            x2_tot.append(((im / 255.0) ** 2).reshape(-1, 3).mean(0))

            # write data
            im = cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{index}_{idx:04d}.png', im)
            m = cv2.imencode('.png', m)[1]
            mask_out.writestr(f'{index}_{idx:04d}.png', m)

# image stats
img_avr = np.array(x_tot).mean(0)
img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
print('mean:', img_avr, ', std:', img_std)