import mnist                                    # 直接使用内置mnist数据集
import numpy as np
from matplotlib.widgets import Button           # 交互作用的按钮
import matplotlib.pyplot as plt                 # 显示图片使用
from PIL import Image                           # 存储图片使用

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文字符
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示正负号


"""
-------->tips:请简要阅读

--》使用前请检查文件路径，以免出现不可预知问题

--》主函数在最底部，如果您想更换需要识别的图片，请在注释部分将  --》img《-- 后面的 --》getimg函数《-- 注释取消并输入合适的地址以便读取图片文件，默认使用本地测试集第一张图片

--》如果您想要使用自己的数据集进行训练，请在  --》learning《-- 文件夹下添加您的数据集，并在 --》type.txt《-- 文件中以 --》 文件名+【空格】+类别 《--方式写入文件名和分类以学习
    否则可能出现不可预知的问题

--》如果在点击按钮时出现了不可预知的bug，请关闭并重新开启（测试过程中出现过按钮只能点击一次的情况）



"""




'''-----------------------------------------------------------------------------------卷积层'''
class Conv3x3:
    # 卷积

    def __init__(self, num_filters):
        self.last_input = None
        self.num_filters = num_filters

        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        使用有效的填充生成所有可能的 3x3 图像区域。
        - image是一个2d数字数组
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用

    def forward(self, input):
        '''
        使用给定的输入执行conv层的前向传递。
        返回一个3d numpy数组，维度为(h, w, num_filters)。
        - input是一个2d numpy数组
        '''
        self.last_input = input
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        # input: 28x28
        # output: 26x26x8
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # 卷积运算，点乘再相加，ouput[i, j] 为向量，8 层
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        # 最后将输出数据返回，便于下一层的输入使用
        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行conv层的向后传递。
        - d_L_d_out是该层输出的损耗梯度。
        —learn_rate为浮点数。
        '''
        # 初始化一组为 0 的 gradient，3x3x8
        d_L_d_filters = np.zeros(self.filters.shape)

        # im_region，一个个 3x3 小矩阵
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters
        return None



'''-----------------------------------------------------------------------------------池化层'''
class MaxPool2:
    # 池化

    def __init__(self):
        self.last_input = None

    def iterate_regions(self, image):
        '''
        生成不重叠的2x2图像区域用于池。
        - image是一个2d numpy数组
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def backprop(self, d_L_d_out):
        '''
        执行maxpool层的向后传递。
        返回该层输入的损耗梯度。
        - d_L_d_out是该层输出的损耗梯度。
        '''
        # 池化层输入数据，26x26x8，默认初始化为 0
        d_L_d_input = np.zeros(self.last_input.shape)

        # 每一个 im_region 都是一个 3x3x8 的8层小矩阵
        # 修改 max 的部分，首先查找 max
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            # 获取 im_region 里面最大值的索引向量，一叠的感觉
            amax = np.amax(im_region, axis=(0, 1))

            # 遍历整个 im_region，对于传递下去的像素点，修改 gradient 为 loss 对 output 的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

    def forward(self, input):
        '''
        使用给定的输入执行maxpool层的前向传递。
        返回一个3d numpy数组，维度为(h / 2, w / 2, num_filters)。
        - input是一个3d numpy数组，维度为(h, w, num_filters)
        '''
        self.last_input = input
        # input: 卷基层的输出，池化层的输入
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output



'''-----------------------------------------------------------------------------------softmax输出层'''
class Softmax:
    # softmax函数

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: 输入层的节点个数，池化层输出拉平之后的
        # nodes: 输出层的节点个数，本例中为 10
        # 构建权重矩阵，初始化随机数，不能太大
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        使用给定的输入执行softmax层的前向传递。
        返回一个包含各自概率值的1d numpy数组。
        - input可以是任意维度的数组。
        '''

        # NEW ADD，13x13x8
        self.last_input_shape = input.shape

        input = input.flatten()

        # NEW ADD, 向量，1352
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        # NEW ADD，softmax 前的向量，10
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行softmax层的向后传递。
        返回该层输入的损耗梯度。
        - d_L_d_out是该层输出的损耗梯度。
        —learn_rate为浮点数
        '''
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # NEW ADD
            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            # 将矩阵从 1d 转为 3d
            # 1352 to 13x13x8
            return d_L_d_inputs.reshape(self.last_input_shape)




'''-----------------------------------------这里开始是和神经网络无关的函数------------------------------------------初始化'''
# 直接选取本地mnist数据集中的1000个数据
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10



'''-----------------------------------------------------------------------------------获取一张图片'''
def get_img(road):
    print("您选择读取一张自己的图片！正在处理...")
    img = Image.open(road)
    return img



'''-----------------------------------------------------------------------------------存储一张图片'''
def save_img(img,road,name):
    im_array = np.array(img)
    saveimg = Image.fromarray(np.uint8(im_array))
    storename=road+"/"+name+".jpg"
    saveimg.save(storename)
    return



'''-----------------------------------------------------------------------------------展示函数，负责输出可视化'''
def show_img(name ,type):

    print("向您展示选择的图片 :")
    img=name
    fig = plt.figure("验证图片")
    plt.clf()

    ax = fig.add_subplot()
    ax.imshow(img)
    ax.set_title('您选择的验证的图片如下,程序将其分类为'+str(type))#给图片加titile

    ax_normal = plt.axes([0.05, 0.01, 0.4, 0.075])
    btn_normal = Button(ax_normal, '开始训练并重置',color='lightblue')
    btn_normal.on_clicked(do_train)

    ax_normal2 = plt.axes([0.55, 0.01, 0.4, 0.075])
    btn_normal2 = Button(ax_normal2, '自设数据集使用[请先观看tips]', color='lightblue')
    btn_normal2.on_clicked(change_train)

    plt.show()
    return



'''-----------------------------------------------------------------------------------前向传播，为dotrain函数提供参数'''
def forward(image, label):
    '''
    完成一个向前通过的CNN和计算的准确性和叉的损失。
    - image是一个2d numpy数组
    - label为参数
    '''
    # 映射到【-0.5，0.5】
    # 前向传播
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # 损失函数和分类是否正确
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc
    # out: vertor of probability
    # loss: num
    # acc: 1 or 0



'''-----------------------------------------------------------------------------------测试输入函数的分类'''
def test_now_img (img):
    image=np.array(img)
    out = conv.forward((image / 255) - 0.5)     #数据的展示
    out = pool.forward(out)
    out = softmax.forward(out)
    #print(out)             #测试各个类的softmax输出
    arr = np.array(out)
    max = np.argmax(arr)    #读取到数据最大的值
    return max              #输出分类的结果



'''-----------------------------------------------------------------------------------学习函数，学习三遍，默认为使用本地mnist数据集'''
def do_train(event):
    train_images = mnist.train_images()[:1000]
    train_labels = mnist.train_labels()[:1000]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]
    for epoch in range(3):
        print("--- 当前循环 %d/3 ---" % (epoch + 1))
        # 初始化参数
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        loss = 0
        num_correct = 0

        # i: index
        # im: image
        # label: label
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i > 0 and i % 100 == 99:
                print(
                    "[已计算 %d] 之前的 100 步中: 平均损失 %.3f | 准确率: %d%%" %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0
            l, acc = train(im, label)
            loss += l
            num_correct += acc

    # 测试结果
    print("\n--- 网络学习结果展示 ---")
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    print("测试损失:", loss / num_tests)
    print("测试准确度:", num_correct / num_tests)
    print("\n学习结束！")
    show_img(img,test_now_img(img))
    return



'''-----------------------------------------------------------------------------------训练函数'''
def train(im, label, lr=.005):
    '''
    完成给定图像和标签的完整训练步骤。
    返回交叉熵损失和精度。
    - image是一个2d numpy数组
    - label为参数
    —lr:学习率
    '''
    # 前向传播
    out, loss, acc = forward(im, label)

    # 梯度计算
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # 反向传播
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc



'''-----------------------------------------------------------------------------------使用自定义的训练集'''
def change_train(event):
    f=open('../模式识别大作业/learnimg/type.txt')
    txt = []
    train_labels_new=[]
    for line in f:
        txt.append(line.strip())
    train_images = mnist.train_images()[:len(txt)]
    for i in range(len(txt)):
        str=txt[i]
        picname=""
        for i2 in range(len(str)):
            if str[i2]==' ':
                print(picname)
                newimg=get_img('../模式识别大作业/learnimg/'+picname)
                num=np.array(newimg)
                #print("newimg",num)
                train_images[i]=num
                picname=""
                continue
            picname+=str[i2]
        print(picname)
        train_labels_new.append(int(picname))
        picname=""
    train_labels=train_labels_new[:len(txt)]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]
    print("使用了新的训练数据集！")
    for epoch in range(3):
        print("--- 当前循环 %d/3 ---" % (epoch + 1))
        # 初始化参数
        '''用来随机排列数组，这里不需要'''
        #permutation = np.random.permutation(len(train_images))
        #train_images = train_images[permutation]
        #train_labels = train_labels[permutation]

        loss = 0
        num_correct = 0

        # i: index
        # im: image
        # label: label
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i > 0 and i % 100 == 99:
                print(
                    "[已计算 %d] 之前的 100 步中: 平均损失 %.3f | 准确率: %d%%" %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0
            l, acc = train(im, label)
            loss += l
            num_correct += acc

    # 测试结果
    print("\n--- 网络学习结果展示 ---")
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = forward(im, label)
        loss += l
        num_correct += acc
    num_tests = len(test_images)
    print("测试损失:", loss / num_tests)
    print("测试准确度:", num_correct / num_tests)
    print("\n学习结束！")
    show_img(img, test_now_img(img))
    return











'''--------------------------------------important！这里开始是主程序！-------------------------------------------------------'''
#这里开始是主程序的部分了


print("网络初始化!")


img=test_images[0]
'''需要使用别的图片测试学习结果请解除下面的注释哦↓'''
#img=get_img('C:/Users/wanderinghunter/Desktop/模式识别大作业/img/newimg.jpg')


show_img(img,test_now_img(img))


print("计算结束！")
