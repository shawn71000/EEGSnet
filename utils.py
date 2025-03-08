import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt

# save as txt
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 二分类混淆矩阵
def perf_ana(lable_true, lable_pred, num_classes):
    cm = confusion_matrix(lable_true, lable_pred)
    n = cm.sum().item()
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP + FN + FP + TN)
    F1score = (2 * Precision * Recall) / (Precision + Recall)
    Specificity = TN / (TN + FP)

    Total = (TP + FN + FP + TN)
    Class1_C = (TP + FP)
    Class1_GT = (TP + FN)
    Class2_C = (TN + FN)
    Class2_GT = (FP + TN)
    ExpAcc = (((Class1_C * Class1_GT) / Total) + ((Class2_C * Class2_GT) / Total)) / Total
    Kappa = (Accuracy - ExpAcc) / (1 - ExpAcc)
    return Recall, Precision, Accuracy, F1score


# k-fold 分割
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    # X = torch.from_numpy(np.array(X))
    # y = torch.from_numpy(np.array(y))
    fold_size = X.shape[0] // k  # 双斜杠表示除完后再向下取整
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 多分类
class ConfusionMatrix(object):

    def __init__(self, num_classes: int, args):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        # self.labels = labels  # 类别标签
        self.args = args

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        # print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 5)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        # table = PrettyTable()  # 创建一个表格
        # table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 5) if TN + FP != 0 else 0.
            F1score = round((2 * Precision * Recall) / (Precision + Recall), 5) if Precision + Recall != 0 else 0.

            # table.add_row([self.labels[i], Precision, Recall, Specificity])
        # print(table)
        return acc, F1score, Recall, Precision

    def plot(self, labels, path_name, fold):  # 绘制混淆矩阵
        matrix = self.matrix
        n = np.sum(matrix)
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Oranges)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), labels)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), labels)
        # 显示colorbar
        # plt.colorbar()
        # plt.xlabel('True Labels')
        # plt.ylabel('Predicted Labels')
        # plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / n * 100
        for x in range(self.num_classes):
            n_label = np.sum(self.matrix[:, x])
            print('n_label: ', n_label)
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x]) / n_label * 100
                plt.text(x, y, str(round(info, 2)) + '%',
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}.png'.format(path_name, 'CM_' + self.args.dataset + str(fold) + '_intra')) # 将混淆矩阵图片保存下来
        plt.close()

    # 这是一个多分类问题（三分类），可以在一张图上绘制多条ROC曲线
    def paint_ROC(self, label, y_score, path_name, fold):
        '''画ROC曲线'''
        plt.figure()
        # 修改颜色
        colors = ['darkred', 'darkorange', 'cornflowerblue']
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(label[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["mean"], tpr["mean"], _ = roc_curve(label.ravel(), y_score.ravel())
        roc_auc["mean"] = auc(fpr["mean"], tpr["mean"])
        lw = 2
        plt.plot(fpr["mean"], tpr["mean"],
                 label='average, ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["mean"]),
                 color='darkgoldenrod', linewidth=lw)
        for i in range(self.num_classes):
            auc_v = roc_auc[i]
            # 输出不同类别的FPR\TPR\AUC
            print('label: {}, fpr: {}, tpr: {}, auc: {}'.format(i, np.mean(fpr[i]), np.mean(tpr[i]), auc_v))
            plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=':', lw=lw,
                     label='Label = {0}, ROC curve (area = {1:0.2f})'.format(i, auc_v))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        plt.grid(linestyle='-.')
        plt.grid(True)
        plt.legend(loc="lower right")
        # plt.show()
        # 保存绘制好的ROC曲线
        plt.savefig('{}/{}.png'.format(path_name, 'ROC_'+self.args.dataset+str(fold)+'_intra'))
        plt.close()
