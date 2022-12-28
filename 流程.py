# 该代码是进行的测试集t10k-images-idx3-ubyte和t10k-labels-idx1-ubyte的图像提取
# 对于训练集train-labels-idx1-ubyte和train-labels-idx1-ubyte，进行相应的替换就行
from torch.utils.data import DataLoader
import weisimin
# 1.解压FashionMNIST
# weisimin.UnpackFashionMNIST('D:\\Office\\Python\\pytorch\\data\\FashionMNIST\\raw\\train-images-idx3-ubyte',
#                             'D:\\Office\\Python\\pytorch\\data\\FashionMNIST\\raw\\train-labels-idx1-ubyte',
#                             'D:\\Office\\Python\\pytorch\\data\\FashionMNIST\\训练')

# 2.生成csv
#weisimin.MakeCsv('D:\\Office\\Python\\pytorch\\data\\FashionMNIST\\测试','测试.csv')

# 3.Softmax
#weisimin.Softmax()

# 4.感知机
# weisimin.perceptron()

# 5.LeNet_5
weisimin.LeNet_5()