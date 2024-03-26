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
from torch.utils.data import TensorDataset, random_split

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

# 定义旋转矩阵
def rotation_matrix(psi):
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0,            0,           1]])
    return Rz

# 修改后的辅助函数，适用于NumPy数组
def extract_data_by_indices(data, indices):
    return data[indices]

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

def transform_data(data_in,mean_in,std_in,Data_frequency,Input_dim,ts):
    data_in = ((data_in.to_numpy() - mean_in) / std_in)
    x= []
    for i in range(0, len(data_in) - Data_frequency * ts, Data_frequency):
        x_i = data_in[i : i + Data_frequency * (ts+1), :].copy()
        x.append(x_i)
    x_arr = np.array(x).reshape(-1, 1, Data_frequency * (ts+1), Input_dim)
    # x_var = Variable(torch.from_numpy(x_arr).float()).to(device)
    return x_arr

def build_Odo_dataset(Data_frequency,Input_dim,device):
    ts = 15-1
    data_In1 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0426mate20.txt', sep=' ', header=None)
    data_in1 = data_In1.iloc[:, [1,2,3,4,5,6]]
    data_in1_2, data_in1_3 = change_data_axis(data_in1)
    data_in1_4 = data_in1.copy() * -1
    data_in1_5 = data_in1_2.copy() * -1
    data_in1_6 = data_in1_3.copy() * -1
    data_Out1 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0426Atlans.txt', sep=' ', header=None)
    data_out1 = data_Out1.iloc[ts:, [1]]

    # 假设ψ是已知的，这里用一个示例角度，你需要用实际的角度来替换这里的0
    psi = np.radians(30)  # 替换为实际的角度值
    Rz = rotation_matrix(psi)
    # 应用旋转矩阵
    # transformed_data = data_in1.apply(lambda row: np.concatenate((Rz.dot(row[:3]), Rz.dot(row[3:6]))), axis=1)
    # 如果你想将结果转换回pandas的DataFrame格式
    # data_in1_tr = pd.DataFrame(transformed_data.tolist(), columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

    data_In2 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0427mate20.txt', sep=' ', header=None)
    data_in2 = data_In2.iloc[:, [1,2,3,4,5,6]]
    data_in2_2, data_in2_3 = change_data_axis(data_in2)
    data_in2_4 = data_in2.copy() * -1
    data_in2_5 = data_in2_2.copy() * -1
    data_in2_6 = data_in2_3.copy() * -1
    data_Out2 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0427Atlans.txt', sep=' ', header=None)
    data_out2 = data_Out2.iloc[:, [1]]

    # transformed_data = data_in2.apply(lambda row: np.concatenate((Rz.dot(row[:3]), Rz.dot(row[3:6]))), axis=1)
    # data_in2_tr = pd.DataFrame(transformed_data.tolist(), columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])


    data_In3 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc0428mate20.txt', sep=' ', header=None)
    data_in3 = data_In3.iloc[:, [1,2,3,4,5,6]]
    data_in3_2, data_in3_3 = change_data_axis(data_in3)
    data_in3_4 = data_in3.copy() * -1
    data_in3_5 = data_in3_2.copy() * -1
    data_in3_6 = data_in3_3.copy() * -1
    data_Out3 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn0428Atlans.txt', sep=' ', header=None)
    data_out3 = data_Out3.iloc[ts:, [1]]

    data_In4 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc1229Mate20Mansour1.txt', sep=' ', header=None)
    data_in4 = data_In4.iloc[:, [1,2,3,4,5,6]]
    data_Out4 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn1229Atlans1.txt', sep=' ', header=None)
    data_out4 = data_Out4.iloc[ts:, [1]]

    data_In5 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\SensorDataForMaplc1229Mate20Mansour2.txt', sep=' ', header=None)
    data_in5 = data_In5.iloc[:, [1,2,3,4,5,6]]
    data_in5_2, data_in5_3 = change_data_axis(data_in5)
    data_in5_4 = data_in5.copy() * -1
    data_in5_5 = data_in5_2.copy() * -1
    data_in5_6 = data_in5_3.copy() * -1
    data_Out5 = pd.read_csv('D:\PhDStudy\pythonproject\DGPS\RNN\odoLearn1229Atlans2.txt', sep=' ', header=None)
    data_out5 = data_Out5.iloc[ts:, [1]]


    eval_size_in = int(len(data_in2) * 0.5)
    full_data_in = pd.concat([data_in1, data_in1_2, data_in1_3, data_in1_4, data_in1_5, data_in1_6,
                             data_in2.iloc[0:eval_size_in],data_in2_2.iloc[0:eval_size_in],data_in2_3.iloc[0:eval_size_in],
                             data_in2_4.iloc[0:eval_size_in],data_in2_5.iloc[0:eval_size_in],data_in2_6.iloc[0:eval_size_in],
                              data_in3,data_in3_2,data_in3_3,data_in3_4, data_in3_5, data_in3_6,
                              data_in5,data_in5_2,data_in5_3,data_in5_4, data_in5_5, data_in5_6])
    test_data_in = data_in2.iloc[eval_size_in:]
    # test_data_in = data_in3
    eval_size_out = int(len(data_out2) * 0.5)
    full_data_out = pd.concat([data_out1] * 6 + [data_out2.iloc[0:eval_size_out]] * 6 + [data_out3] * 6 + [data_out5] * 6)
    test_data_out = data_out2.iloc[eval_size_out+ts:]
    # test_data_out = data_out3

    train_arrT_in = full_data_in.to_numpy()
    mean_in = train_arrT_in.mean(axis=0)
    std_in = train_arrT_in.std(axis=0)
    train_arrT_out = full_data_out.to_numpy()
    max_out = train_arrT_out.max(axis=0)
    min_out = train_arrT_out.min(axis=0)

    data_in1 = transform_data(data_in1, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in1_2 = transform_data(data_in1_2, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in1_3 = transform_data(data_in1_3, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in1_4 = transform_data(data_in1_4, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in1_5 = transform_data(data_in1_5, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in1_6 = transform_data(data_in1_6, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2 = transform_data(data_in2.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2_2 = transform_data(data_in2_2.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2_3 = transform_data(data_in2_3.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2_4 = transform_data(data_in2_4.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2_5 = transform_data(data_in2_5.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in2_6 = transform_data(data_in2_6.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3 = transform_data(data_in3, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3_2 = transform_data(data_in3_2, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3_3 = transform_data(data_in3_3, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3_4 = transform_data(data_in3_4, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3_5 = transform_data(data_in3_5, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in3_6 = transform_data(data_in3_6, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in4 = transform_data(data_in4, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5 = transform_data(data_in5, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5_2 = transform_data(data_in5_2, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5_3 = transform_data(data_in5_3, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5_4 = transform_data(data_in5_4, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5_5 = transform_data(data_in5_5, mean_in, std_in, Data_frequency, Input_dim, ts)
    data_in5_6 = transform_data(data_in5_6, mean_in, std_in, Data_frequency, Input_dim, ts)
    # test_arr_in = torch.cat([data_in4, data_in5], dim=0)

    # data_in1_tr = transform_data(data_in1_tr, mean_in, std_in, Data_frequency, Input_dim, ts)
    # data_in2_tr = transform_data(data_in2_tr.iloc[0:eval_size_in], mean_in, std_in, Data_frequency, Input_dim, ts)

    # full_arr_in = torch.cat([data_in1, data_in1_2, data_in1_3, data_in1_4, data_in1_5, data_in1_6,
    #                          data_in2,data_in2_2,data_in2_3,data_in2_4,data_in2_5,data_in2_6,
    #                          data_in3,data_in3_2,data_in3_3,data_in3_4,data_in3_5,data_in3_6,
    #                          data_in5,data_in5_2,data_in5_3,data_in5_4,data_in5_5,data_in5_6], dim=0)

    # full_arr_in = torch.cat([data_in1, data_in1_tr, data_in2, data_in2_tr], dim=0)
    test_arr_in = transform_data(test_data_in, mean_in, std_in, Data_frequency, Input_dim, ts)
    test_arr_in = Variable(torch.from_numpy(test_arr_in).float()).to(device)
    # full_data_out2 = pd.concat([data_out1] * 2 + [data_out2.iloc[0:eval_size_out]] * 2)
    # full_arr_out = (full_data_out2.to_numpy() - min_out) / (max_out - min_out)
    test_arr_out = (test_data_out.to_numpy() - min_out) / (max_out - min_out)
    #
    # full_arr_out = Variable(torch.from_numpy(full_arr_out).float()).to(device)
    test_arr_out = Variable(torch.from_numpy(test_arr_out).float()).to(device)
    #
    # full_dataset = torch.utils.data.TensorDataset(full_arr_in, full_arr_out)
    test_dataset = torch.utils.data.TensorDataset(test_arr_in, test_arr_out)
    #
    # train_size = int(len(full_arr_in) * 0.7)
    # val_size = len(full_arr_in) - train_size

    # 对每个数据集单独分割并提取相应的输出数据
    # datasets = [data_in1, data_in1_tr, data_in2, data_in2_tr]
    # outputs = [data_out1, data_out1, data_out2.iloc[0:eval_size_out], data_out2.iloc[0:eval_size_out]]
    datasets = [data_in1, data_in2, data_in4, data_in5]
    outputs = [data_out1, data_out2.iloc[ts:eval_size_out], data_out4, data_out5]
    # 转换numpy数组到torch张量
    dataset_tensors = [torch.tensor(d).float().to(device) for d in datasets]  # 这假设你的输入数据已经是适合的数值类型

    # 这里不需要更改outputs，因为它们已经是DataFrame格式

    train_inputs = []
    train_outputs = []
    val_inputs = []
    val_outputs = []

    for data, output in zip(dataset_tensors, outputs):
        train_size = int(len(data) * 0.7)
        val_size = len(data) - train_size
        train_subset, val_subset = random_split(data, [train_size, val_size])

        # 提取训练输入和输出
        train_indices = train_subset.indices
        train_inputs.append(data[train_indices])
        train_outputs.append(extract_data_by_indices(output.to_numpy(), train_indices))  # 用to_numpy()确保是NumPy数组

        # 提取验证输入和输出
        val_indices = val_subset.indices
        val_inputs.append(data[val_indices])
        val_outputs.append(extract_data_by_indices(output.to_numpy(), val_indices))

    # 合并训练和验证的输入输出，注意这里是Tensor操作
    full_train_in = torch.cat(train_inputs, dim=0)
    full_train_out = torch.tensor(np.concatenate(train_outputs)).float()
    full_train_out = (full_train_out - min_out) / (max_out - min_out)
    full_train_out = full_train_out.float().to(device)

    full_val_in = torch.cat(val_inputs, dim=0)
    full_val_out = torch.tensor(np.concatenate(val_outputs)).float()
    full_val_out = (full_val_out - min_out) / (max_out - min_out)
    full_val_out = full_val_out.float().to(device)

    # 创建TensorDataset
    full_train_dataset = TensorDataset(full_train_in, full_train_out)
    full_val_dataset = TensorDataset(full_val_in, full_val_out)

    # return full_dataset,test_dataset,train_size,val_size,max_out,min_out
    return full_train_dataset, full_val_dataset, test_dataset, max_out, min_out
