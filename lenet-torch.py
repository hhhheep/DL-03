import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2 as cv
from top_1_top_5 import confusion_matrix
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")## .to(device)
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


train_dataset = class_img_data(file_name="train")
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=300, shuffle=True)

vaild_dataset = class_img_data(file_name="val")
vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=500, shuffle=True)

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_total_steps = len(data_loader)




for epoch in range(num_epochs):

    epoch_acc = []
    epoch_loss = []

    for i, (batch, labels) in tqdm(enumerate(data_loader)):
        '''
        
        print loss acc
        儲存 loss acc
        儲存權重
         
        '''
        batch = batch.clone().detach().to(torch.float32).to(device)
        labels = labels.to(device)


        pred_y = model(batch)
        _, predicted = torch.max(pred_y, 1)

        loss = criterion(pred_y,labels)
        loss.backward()

        acc = np.diag(confusion_matrix(predicted.to("cpu").numpy(),labels.to("cpu").numpy())).sum()/labels.shape[0]

        epoch_acc.append(acc)
        epoch_loss.append(loss)

        optimizer.step()
        optimizer.zero_grad()

        if i % 2 == 0:
            print('epoch: %d batch: %d acc: %.2f loss: %.4f' % (epoch, i, acc, loss))

    L = sum(epoch_loss)/i
    A = sum(epoch_acc)/i

    with torch.no_grad():

        model.eval()
        for i, (v_batch, v_labels) in enumerate(data_loader):

            v_batch = v_batch.clone().detach().to(torch.float32).to(device)
            v_labels = v_labels.to(device)

            vail_y = model(v_batch)
            _, predicted = torch.max(vail_y, 1)
            v_acc = np.diag(confusion_matrix(predicted.to("cpu").numpy(), v_labels.to("cpu").numpy())).sum() / v_labels.shape[0]

            v_loss = criterion(vail_y, v_labels)
            break

    path = 'epoch_result_th.csv'
    with open(path, 'a') as f:
        f.write('{0},{1},{2},{3}'.format(round(L.to("cpu").item(),4), round(A.to("cpu").item(),4), round(v_loss.to("cpu").item(),4),round(v_acc.to("cpu").item(),4)))
    f.close()

    if epoch == 3 :
        folder_name = "models_th"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.save(model, os.path.join(folder_name, "epoch_{0}_model.pth".format(epoch)))


