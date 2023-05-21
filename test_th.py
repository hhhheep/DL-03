import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2 as cv
from top_1_top_5 import confusion_matrix
# import os
from tqdm import tqdm
from thop import profile

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")## .to(device)
# device = "cpu"
num_epochs = 300


class class_img_data(Dataset):

    def __init__(self,file_name):
        self.path = ".\\images\\"
        self.train_data = pd.read_csv(self.path + str(file_name) + ".txt", header=None, sep=" ")
        self.train_data.columns = ["image", "label"]
        self.img_path_l = self.train_data["image"]
        self.img_label = self.train_data["label"]

    def __len__(self):
        return len(self.train_data["image"])

    def __getitem__(self, idx):
        image_path = self.img_path_l[idx]
        label = self.img_label[idx]
        image = cv.imread(self.path + str(image_path))
        image = cv.resize(image,(64,64))
        image = image.astype(np.float32)/255.0
        tensor_image = torch.Tensor(image)
        tensor_image = tensor_image.transpose(0,2)

        return tensor_image, label

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.LeNet5_cov = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.Sigmoid(),
            nn.BatchNorm2d(6),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2, 2))
        )
            # nn.view(-1, self.num_flat_features()),
        self.LeNet5_Lin = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(84, 50),
            nn.Softmax(dim=-1)
        )
    def forward(self,x):

        x = self.LeNet5_cov(x)
        # print(x.shape)
        x = x.view(x.shape[0],-1)
        x = self.LeNet5_Lin(x)

        return x

model = Lenet5().to(device)
torch.load("epoch_297_model.pth", map_location=device)

test_dataset = class_img_data(file_name="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=True)

n_total_steps = len(test_loader)
epoch_acc = []

for i, (batch, labels) in tqdm(enumerate(test_loader)):

    batch = batch.clone().detach().to(torch.float32).to(device)
    labels = labels.to(device)

    pred_y = model(batch)
    _, predicted = torch.max(pred_y, 1)

    acc = np.diag(confusion_matrix(predicted.to("cpu").numpy(), labels.to("cpu").numpy())).sum() / labels.shape[0]

    epoch_acc.append(acc)

acc = sum(epoch_acc)/(i+1)
print(acc)

dummy_input = torch.rand(1, 3, 64, 64).to(device)

print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)


torch.cuda.synchronize()

repetitions = 300

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))


flops, params = profile(model, inputs=(dummy_input,))
print('FLOPs = ' + str(flops/1000) + 'K')
print('Params = ' + str(params/1000) + 'K')













