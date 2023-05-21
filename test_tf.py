import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import os
import cv2
from top_1_top_5 import confusion_matrix



path = os.getcwd()
path = path + "\\images\\"


def read_local_txt(file, path=path):
    train_path = path + str(file) + '.txt'
    file_inf = pd.read_csv(train_path, header=None, sep=" ")
    file_inf.columns = ["image", "label"]
    return file_inf
    # return dataset



def read_image(batch, path=path,size = 64):

    batch = list(batch)
    # if len(batch) > 1:
    images = np.empty([0,size,size,3])
    for img_path in batch:
        img = cv2.imread(path + str(img_path))
        img = cv2.resize(img, (size, size))
        img = img.astype('float32') / 255.0
        # img = cv.imread(path + str(img_path))  # ???修改 路徑格式
        # img = cv.resize(img, (256, 256))
        images = np.append(images, img.reshape(1, size, size, 3), axis=0)

    return images
    # else:
    #     img = cv2.imread(path + str(batch[0].numpy(), 'utf-8'))
    #     img = cv2.resize(img, (64,64))
    #     img = img.astype('float32') / 255.0
    #     img = tf.transpose(img, perm=[2, 0, 1])
    #     return img

def train_batch(file_inf,batch_size = 20):

    if len(file_inf) < batch_size:
        batch_size = len(file_inf)
    batch_inf = file_inf.sample(n=batch_size,replace=False,random_state=123)
    file_inf = file_inf.drop(index = batch_inf.index)

    batch = batch_inf.iloc[:,0]
    label = batch_inf.iloc[:,1]

    return batch,label,file_inf,batch_size

# 加载模型
model = tf.saved_model.load('epoch_267_tf_model.pth')

test_inf = read_local_txt("test")
test_len = test_inf.shape[0]

# 使用加载的模型进行预测

epoch_acc = []

for n in tqdm(range(math.ceil(test_len / 500))):
    # 讀取一個批次的圖片資料
    batch, label, train_inf1, batch_size1 = train_batch(test_inf, batch_size=500)
    train_imges = read_image(batch)
    train_imges = train_imges.astype(np.float32)
    label = label.to_numpy().astype(np.float32)

    predictions = model(train_imges)

    pred_Y = np.argmax(predictions, axis=1)
    acc = sum(np.diag(confusion_matrix(pred_Y.astype(np.int32), label.astype(np.int32)))) / batch_size1
    epoch_acc.append(acc)

acc = sum(epoch_acc)/(n+1)
print(acc)

device = 'GPU:0'
repetitions = 300

dummy_input = tf.random.uniform((1, 64, 64, 3))  # 創建測試輸入

# 預熱，GPU平時可能處於休眠狀態，因此需要預熱
print('預熱中...\n')
for _ in range(100):
    _ = model(dummy_input)

# 同步，等待所有GPU任務處理完才返回CPU主線程
tf.device(device)
with tf.GradientTape(persistent=True):
    _ = model(dummy_input)

# 初始化一個時間容器
timings = np.zeros((repetitions, 1))

print('測試中...\n')
for rep in tqdm(range(repetitions)):
    tf.device(device)
    with tf.GradientTape(persistent=True) as tape:
        starter = tf.timestamp()
        _ = model(dummy_input)
        ender = tf.timestamp()
    tf.device('/CPU:0')  # 切換回CPU設備
    curr_time = (ender - starter) * 1000  # 從starter到ender之間的時間，單位為毫秒
    timings[rep] = curr_time

avg = np.sum(timings) / repetitions
print('\navg={}\n'.format(avg))

