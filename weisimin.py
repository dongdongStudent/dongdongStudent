# 该代码是进行的测试集t10k-images-idx3-ubyte和t10k-labels-idx1-ubyte的图像提取
# 对于训练集train-labels-idx1-ubyte和train-labels-idx1-ubyte，进行相应的替换就行
import numpy as np
import struct
from PIL import Image
import os, glob
import os,glob
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch import nn
from d2l import torch as d2l


def UnpackFashionMNIST(image, label,datas_root1):
    """

    image: 图片位置
    label: 标号文字
    datas_root1: 存储位置

    """
    data_file = image
    fsize = os.path.getsize(data_file)
    data_file_size = fsize
    data_file_size = str(data_file_size - 16) + 'B'

    data_buf = open(data_file, 'rb').read()

    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', data_buf, 0)
    datas = struct.unpack_from('>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)

    label_file = label
    # It's 60008B, but we should set to 60000B
    fsize = os.path.getsize(label_file)
    label_file_size = fsize
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from('>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    # 修改路径
    if not os.path.exists(datas_root1):
        os.mkdir(datas_root1)

    for i in range(10):
        file_name = datas_root1 + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    for ii in range(numLabels):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root1 + os.sep + str(label) + os.sep + 'mnist_test_' + str(ii) + '.png'
        img.save(file_name)

def MakeCsv(datas_root1,CsvName):
    """
    datas_root1:图片路径
    CsvName:

    """
    class_to_num = {}
    class_name_list = os.listdir(datas_root1)

    for class_name in class_name_list:
        class_to_num[class_name] = len(class_to_num.keys())


    image_dir = []
    for class_name in class_name_list:
        a = os.path.join(datas_root1, class_name, '*.png')
        image_dir += glob.glob(a)

    # 写csv
    with open(CsvName, mode='w', newline='') as f:
        writer = csv.writer(f)
        for image in image_dir:
            class_name = image.split(os.sep)[-2]
            label = class_to_num[class_name]
            writer.writerow([image, label])

class makedataset(Dataset):
    #(文件名字,文件大小,)
    def __init__(self,csv_filename,resize):
        super(makedataset, self).__init__()

        self.csv_filename=csv_filename
        self.resize=resize
        self.image,self.label=self.load_csv()


    def load_csv(self):
        image=[]
        label=[]
        with open(self.csv_filename) as f:
            reader = csv.reader(f)
            for row in reader:
                i,l=row
                image.append(i)
                label.append(l)
        return image ,label


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        tf=transforms.Compose([lambda x:Image.open(x).convert('L'),transforms.Resize(self.resize),transforms.ToTensor()])
        image_tensor=tf(self.image[idx])
        label_tensor=torch.tensor(int(self.label[idx]))
        return image_tensor,label_tensor

# Softmax
#(数据,图片尺寸,)

def Softmax():
    train_db=makedataset('训练.csv', 28 )
    test_db=makedataset('测试.csv', 28)
    train_iter = DataLoader(dataset=train_db,batch_size=256,shuffle=True)
    test_iter = DataLoader(dataset=test_db,batch_size=256,shuffle=True)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights);

    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs = 1
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    # 7.开始训练(比较模型,训练参数,测试参数,损失函数,循环周期,更新)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()


def perceptron():
    # 2. 激活函数
    def relu(X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    # 3. 模型
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
        return (H@W2 + b2)

    # 4. 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 5. 训练
    train_db = makedataset('训练.csv', 28)
    test_db = makedataset('测试.csv', 28)
    train_iter = DataLoader(dataset=train_db, batch_size=1000, shuffle=True)
    test_iter = DataLoader(dataset=test_db, batch_size=1000, shuffle=True)

    # 6. 设置参数
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) #(784,256)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)#(256,10)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        # 6.1 设置参数params
    params = [W1, b1, W2, b2]
        # 6.2 lr(步长)
    updater = torch.optim.SGD(params, lr=0.1)

    # 7.开始训练(模型,训练参数,测试参数,损失函数,循环周期,)
    d2l.train_ch3(net, train_iter, test_iter, loss, 2, updater)

    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()

def LeNet_5():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)

    # 5. 训练
    train_db = makedataset('训练.csv', 28)
    test_db = makedataset('测试.csv', 28)
    train_iter = DataLoader(dataset=train_db, batch_size=256, shuffle=True)
    test_iter = DataLoader(dataset=test_db, batch_size=256)


    def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(net, nn.Module):
            net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(d2l.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    #@save
    def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
        """用GPU训练模型(在第六章定义)"""
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'test acc'])
        timer, num_batches = d2l.Timer(), len(train_iter)
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，样本数
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (train_l, train_acc, None))
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')


    train_ch6(net, train_iter, test_iter, 2, 0.9, d2l.try_gpu())
    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()