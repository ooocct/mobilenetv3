# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, lmdb, pickle, six
from PIL import Image
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

import pandas as pd
import numpy as np
from torch.autograd import Variable


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform

    def __getitem__(self, idx):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        label = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_LMDB':
        print("reading from datapath", args.data_path)
        path = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(path, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

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
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def change_data_axis(data_in):
    column_0 = data_in.iloc[:, 0]
    column_1 = data_in.iloc[:, 1]
    column_2 = data_in.iloc[:, 2]
    column_3 = data_in.iloc[:, 3]
    column_4 = data_in.iloc[:, 4]
    column_5 = data_in.iloc[:, 5]
    data_in_1 = data_in.copy()
    data_in_1.iloc[:, 0] = column_2
    data_in_1.iloc[:, 1] = column_0
    data_in_1.iloc[:, 2] = column_1
    data_in_1.iloc[:, 3] = column_5
    data_in_1.iloc[:, 4] = column_3
    data_in_1.iloc[:, 5] = column_4
    data_in_2 = data_in.copy()
    data_in_2.iloc[:, 0] = column_1
    data_in_2.iloc[:, 1] = column_2
    data_in_2.iloc[:, 2] = column_0
    data_in_2.iloc[:, 3] = column_4
    data_in_2.iloc[:, 4] = column_5
    data_in_2.iloc[:, 5] = column_3
    return data_in_1,data_in_2

def transform_data(data_in,mean_in,std_in,Data_frequency,Input_dim,device):
    data_in = ((data_in.to_numpy() - mean_in) / std_in)
    x= []
    for i in range(0, len(data_in) - Data_frequency, Data_frequency):
        x_i = data_in[i : i + Data_frequency*2, :].copy()
        x.append(x_i)
    x_arr = np.array(x).reshape(-1, 1, Data_frequency*2, Input_dim)
    x_var = Variable(torch.from_numpy(x_arr).float()).to(device)
    return x_var

def build_Odo_dataset(Data_frequency,Input_dim,device):
    data_In1 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0426mate20.txt', sep=' ', header=None)
    data_in1 = data_In1.iloc[:, [1,2,3,4,5,6]]
    data_in1_2, data_in1_3 = change_data_axis(data_in1)
    data_in1_4 = data_in1.copy() * -1
    data_in1_5 = data_in1_2.copy() * -1
    data_in1_6 = data_in1_3.copy() * -1
    data_Out1 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0426Atlans.txt', sep=' ', header=None)
    data_out1 = data_Out1.iloc[1:, [1]]

    data_In2 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0427mate20.txt', sep=' ', header=None)
    data_in2 = data_In2.iloc[:, [1,2,3,4,5,6]]
    data_in2_2, data_in2_3 = change_data_axis(data_in2)
    data_in2_4 = data_in2.copy() * -1
    data_in2_5 = data_in2_2.copy() * -1
    data_in2_6 = data_in2_3.copy() * -1
    data_Out2 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0427Atlans.txt', sep=' ', header=None)
    data_out2 = data_Out2.iloc[1:, [1]]

    data_In3 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0428mate20.txt', sep=' ', header=None)
    data_in3 = data_In3.iloc[:, [1,2,3,4,5,6]]
    data_in3_2, data_in3_3 = change_data_axis(data_in3)
    data_in3_4 = data_in3.copy() * -1
    data_in3_5 = data_in3_2.copy() * -1
    data_in3_6 = data_in3_3.copy() * -1
    data_Out3 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0428Atlans.txt', sep=' ', header=None)
    data_out3 = data_Out3.iloc[1:, [1]]

    data_In4 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc1229Mate20Mansour1.txt', sep=' ', header=None)
    data_in4 = data_In4.iloc[:, [1,2,3,4,5,6]]
    data_Out4 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn1229Atlans1.txt', sep=' ', header=None)
    data_out4 = data_Out4.iloc[1:, [1]]

    data_In5 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc1229Mate20Mansour2.txt', sep=' ', header=None)
    data_in5 = data_In5.iloc[:, [1,2,3,4,5,6]]
    data_in5_2, data_in5_3 = change_data_axis(data_in5)
    data_in5_4 = data_in5.copy() * -1
    data_in5_5 = data_in5_2.copy() * -1
    data_in5_6 = data_in5_3.copy() * -1
    data_Out5 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn1229Atlans2.txt', sep=' ', header=None)
    data_out5 = data_Out5.iloc[1:, [1]]


    eval_size_in = int(len(data_in2) * 0.5)
    full_data_in = pd.concat([data_in1, data_in1_2, data_in1_3, data_in1_4, data_in1_5, data_in1_6,
                             data_in2.iloc[0:eval_size_in],data_in2_2.iloc[0:eval_size_in],data_in2_3.iloc[0:eval_size_in],
                             data_in2_4.iloc[0:eval_size_in],data_in2_5.iloc[0:eval_size_in],data_in2_6.iloc[0:eval_size_in],
                              data_in3,data_in3_2,data_in3_3,data_in3_4, data_in3_5, data_in3_6,
                              data_in5,data_in5_2,data_in5_3,data_in5_4, data_in5_5, data_in5_6])
    # test_data_in = data_in2.iloc[eval_size_in:]
    test_data_in = data_in4
    eval_size_out = int(len(data_out2) * 0.5)
    full_data_out = pd.concat([data_out1] * 6 + [data_out2.iloc[0:eval_size_out]] * 6 + [data_out3] * 6 + [data_out5] * 6)
    # test_data_out = data_out2.iloc[eval_size_out+1:]
    test_data_out = data_out4

    train_arrT_in = full_data_in.to_numpy()
    mean_in = train_arrT_in.mean(axis=0)
    std_in = train_arrT_in.std(axis=0)
    train_arrT_out = full_data_out.to_numpy()
    max_out = train_arrT_out.max(axis=0)
    min_out = train_arrT_out.min(axis=0)

    data_in1 = transform_data(data_in1, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in1_2 = transform_data(data_in1_2, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in1_3 = transform_data(data_in1_3, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in1_4 = transform_data(data_in1_4, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in1_5 = transform_data(data_in1_5, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in1_6 = transform_data(data_in1_6, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2 = transform_data(data_in2.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2_2 = transform_data(data_in2_2.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2_3 = transform_data(data_in2_3.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2_4 = transform_data(data_in2_4.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2_5 = transform_data(data_in2_5.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in2_6 = transform_data(data_in2_6.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3 = transform_data(data_in3, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3_2 = transform_data(data_in3_2, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3_3 = transform_data(data_in3_3, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3_4 = transform_data(data_in3_4, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3_5 = transform_data(data_in3_5, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in3_6 = transform_data(data_in3_6, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5 = transform_data(data_in5, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5_2 = transform_data(data_in5_2, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5_3 = transform_data(data_in5_3, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5_4 = transform_data(data_in5_4, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5_5 = transform_data(data_in5_5, mean_in, std_in, Data_frequency, Input_dim, device)
    data_in5_6 = transform_data(data_in5_6, mean_in, std_in, Data_frequency, Input_dim, device)
    # test_arr_in = torch.cat([data_in4, data_in5], dim=0)

    full_arr_in = torch.cat([data_in1, data_in1_2, data_in1_3, data_in1_4, data_in1_5, data_in1_6,
                             data_in2,data_in2_2,data_in2_3,data_in2_4,data_in2_5,data_in2_6,
                             data_in3,data_in3_2,data_in3_3,data_in3_4,data_in3_5,data_in3_6,
                             data_in5,data_in5_2,data_in5_3,data_in5_4,data_in5_5,data_in5_6], dim=0)
    test_arr_in = transform_data(test_data_in, mean_in, std_in, Data_frequency, Input_dim, device)

    full_arr_out = (full_data_out.to_numpy() - min_out) / (max_out - min_out)
    test_arr_out = (test_data_out.to_numpy() - min_out) / (max_out - min_out)

    full_arr_out = Variable(torch.from_numpy(full_arr_out).float()).to(device)
    test_arr_out = Variable(torch.from_numpy(test_arr_out).float()).to(device)

    full_dataset = torch.utils.data.TensorDataset(full_arr_in, full_arr_out)
    test_dataset = torch.utils.data.TensorDataset(test_arr_in, test_arr_out)

    train_size = int(len(full_arr_in) * 0.7)
    val_size = len(full_arr_in) - train_size

    return full_dataset,test_dataset,train_size,val_size,max_out,min_out
