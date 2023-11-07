import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# autumn data
import torch
import torch.nn as nn
import pickle
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from collections import defaultdict
import pdb
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torchaudio

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class AudioCSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: Path,
        randomize: bool = True,
        apply_padding: bool = False,
        is_train: bool = True,
        sample_rate: int = 16000,
        seed: int = 42,
        is_test: bool = False
    ):
        self.is_train = is_train
        self.spectrogram_extractor = Spectrogram(n_fft=512, hop_length=160, 
            win_length=512, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=16000, n_fft=512, 
            n_mels=112, fmin=50, fmax=14000, ref=1.0, 
            amin=1e-10, top_db=None, freeze_parameters=True)
        self.target_T = 448
        self.files, self.labels = self._parse_csv(csv_path)
        self.randomize = randomize
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.apply_padding = apply_padding
        self.class_to_indices = self._get_class_indices()
        self.freq_ratio = 2
        self.crop_pts = 72000
        self.is_test = is_test

    @staticmethod
    def _parse_csv(csv_path):
        df = pd.read_csv(csv_path)
        return df["hdd_path"].tolist(), df["label"].tolist()
    
    # To get dataset size
    def _get_class_indices(self):
        class_to_indices = defaultdict(list)
        for i, target in enumerate(self.labels):
            class_to_indices[target].append(i)
        return class_to_indices

    def get_spectrogram(self, y):
        s = self.spectrogram_extractor(y).squeeze(0) # [1, 1, T, F] -> [1, T, F]
        s = self.logmel_extractor(s)        # [1, T, F] -> [1, T, F]
        return s
    
    def _zero_padding(self, s):
        # s: [1, T, F]
        if s.shape[1] <= self.target_T: # 448보다 작을 때, 4.48초보다 짧을 때
            index = self.rng.integers(0, self.target_T - s.shape[1] + 1) if self.randomize else 0
            zero_signal = np.ones((s.shape[0], self.target_T, s.shape[2])) * (-100.0)
            zero_signal[:, index:index + s.shape[1], :] = s
            zero_signal = torch.FloatTensor(zero_signal)
        else:
            index = self.rng.integers(0, s.shape[1] - self.target_T + 1) if self.randomize else 0
            zero_signal = s[:, index:index+self.target_T, :]
        return zero_signal    

    def _spec_to_img(self, s):

        freq_ratio = 2
        # [C, T, F] --> torch.Size([1, 448, 112])
        C, T, F = s.shape
        if T % 2 == 1:
            s = nn.functional.interpolate(s.unsqueeze(0), (int(s.shape[1]+1), s.shape[2]), mode="bicubic", align_corners=True).squeeze(0)

        # [C, T, F] --> [C, F, T]
        # torch.Size([1, 448, 112]) --> torch.Size([1, 112, 448])
        s = s.permute(0,2,1).contiguous()

        # torch.Size([1, 112, 448]) --> torch.Size([1, 112, 2, 224])
        s = s.reshape(s.shape[0], s.shape[1], freq_ratio, s.shape[2]//freq_ratio)

        # torch.Size([1, 112, 2, 224]) --> torch.Size([1, 2, 112, 224])
        s = s.permute(0,2,1,3).contiguous()

        # torch.Size([1, 2, 112, 224]) --> torch.Size([1, 224, 224])
        s = s.reshape(s.shape[0], s.shape[1] * s.shape[2], s.shape[3])

        return torch.FloatTensor(s)

    def _postprocess_spectrogram(self, s):
        if self.apply_padding:
            s = self._zero_padding(s)
        s = self._spec_to_img(s)
        return torch.FloatTensor(s)
    
    def _crop_wav(self, y):
        # [1, raw audio]
        # 1, 73000이면 0~1000까지여야해        
        if y.shape[1] >= self.crop_pts:
            index = self.rng.integers(0, y.shape[1] - self.crop_pts + 1)
            y = y[:, index:index+self.crop_pts]
        return y
    
    def _transform(self, audiopath):
        y, sr = torchaudio.load(audiopath) # [1, raw audio]
        assert sr == 16000, "Sample rate error"
        if self.apply_padding:
            y = self._crop_wav(y)
        s = self.get_spectrogram(y) # [1, T, F=112]
        return self._postprocess_spectrogram(s) # s: [1, T, F]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        if self.is_test:
            return self._transform(self.files[i]), int(self.labels[i]), self.files[i]
        return self._transform(self.files[i]), int(self.labels[i])


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'autumn':
        root = os.path.join("/home/hdd1/won_hdd/DB/autumn_1014/meta/", 'data_autumn_train_1018.csv' if is_train else 'data_autumn_valid_1018.csv')
        randomize = True
        if is_train:
            randomize = False   
        if args.test:    
            root = os.path.join("/home/hdd1/won_hdd/DB/autumn_1014/meta/", 'data_autumn_valid_1018.csv' if is_train else 'data_autumn_test_1018.csv')     
            dataset = AudioCSVDataset(csv_path=Path(root), randomize=randomize, is_test=True, apply_padding=args.apply_padding, is_train=is_train)
        else:
            dataset = AudioCSVDataset(csv_path=Path(root), randomize=randomize, apply_padding=args.apply_padding, is_train=is_train)
        nb_classes = 3
    elif args.data_set == 'autumn_bg':
        root = os.path.join(args.data_path, "meta", 'data_autumn_train_1018.csv' if is_train else 'data_autumn_valid_1018.csv')
        randomize = True
        if is_train:
            randomize = False            
        dataset = AudioCSVDataset(Path(root), randomize=randomize, 
                                  apply_padding=True, is_train=is_train)
        nb_classes = 4

    return dataset, nb_classes

    
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

