import numpy as np
# from sklearn.metrics import accuracy_score


def confusion_matrix(preds, labels):
    size = 50
    conf_matrix = np.zeros(size*size).reshape((size,size))
    # preds = np.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p-1, t-1] += 1
    return conf_matrix


# y_true为测试集标签，y_pred为模型预测结果，proba为预测概率矩阵
# def top1_accuracy(y_true, proba):
#     # 对预测概率矩阵按照概率值从高到低排序
#     sorted_proba = np.argsort(-proba, axis=1)
#     # 取最有可能的标签作为预测结果
#     y_top1 = sorted_proba[:, 0]
#     # 计算预测准确率
#     return accuracy_score(y_true, y_top1)
#
# def top5_accuracy(y_true, proba):
#     # 对预测概率矩阵按照概率值从高到低排序
#     sorted_proba = np.argsort(-proba, axis=1)
#     # 取前5个最有可能的标签作为预测结果
#     y_top5 = sorted_proba[:, :5]
#     # 判断真实标签是否在预测结果中出现
#     correct_top5 = [y_true[i] in y_top5[i] for i in range(len(y_true))]
#     # 计算预测准确率
#     return sum(correct_top5) / len(correct_top5)

if __name__ == '__main__':
    print(sum(np.diag(confusion_matrix(np.array([1,2,3,2,1,2,3,1,2,3]).reshape(10,1),np.array([3,2,1,2,3,1,2,3,1,2]).reshape(10,1)))))