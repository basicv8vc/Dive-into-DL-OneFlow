# 线性回归的简洁实现

在过去的几年里，出于对深度学习强烈的兴趣，许多公司、学者和业余爱好者开发了各种成熟的开源框架。通过这些框架可以自动化实现基于梯度的学习算法中重复性的工作。
在3.2节中，我们只依赖了：（1）通过张量来进行数据存储和线性代数；（2）通过自动微分来计算梯度。实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用，现代深度学习库也为我们实现了这些组件。

在本节中，我们将介绍如何通过使用深度学习框架来简洁地实现3.2节中的线性回归模型。

## 3.3.1. 生成数据集

与3.2节中类似，我们首先生成数据集。

```python
import oneflow as flow
import numpy as np
from oneflow.utils import data
from utils import *

# 用于DataLoader复现结果
g = flow.Generator()
g.manual_seed(123)

true_w = flow.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

## 3.3.2. 读取数据集

我们可以调用框架中现有的API来读取数据。我们将`features`和`labels`作为API的参数传递，并在实例化数据迭代器对象时指定`batch_size`。此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个OneFlow数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用`data_iter`的方式与我们在3.2节中使用`data_iter`函数的方式相同。为了验证是否正常工作，让我们读取并打印第一个小批量样本。
与3.2节不同，这里我们使用`iter`构造Python迭代器，并使用`next`从迭代器中获取第一项。

```python
next(iter(data_iter))
```
    [tensor([[-0.2061, -0.1628],
            [ 1.0437, -0.1884],
            [ 1.1569,  0.0608],
            [-0.0457,  0.4002],
            [ 1.1895, -0.2745],
            [ 0.2632, -0.2540],
            [-0.0290, -0.4582],
            [ 1.5555, -1.0068],
            [-0.8332,  2.1693],
            [-2.4869,  1.0972]], dtype=oneflow.float32),
    tensor([[ 4.3547],
            [ 6.9433],
            [ 6.2950],
            [ 2.7572],
            [ 7.5172],
            [ 5.5745],
            [ 5.7003],
            [10.7223],
            [-4.8401],
            [-4.5075]], dtype=oneflow.float32)]


## 3.3.3. 定义模型

当我们在3.2节中实现线性回归时，我们明确定义了模型参数变量，并编写了计算的代码，这样通过基本的线性代数运算得到输出。但是，如果模型变得更加复杂，而且当你几乎每天都需要实现模型时，你会想简化这个过程。这种情况类似于从头开始编写自己的博客。做一两次是有益的、有启发性的，但如果每次你每需要一个博客就花一个月的时间重新发明轮子，那你将是一个糟糕的网页开发者。

对于标准操作，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。我们首先定义一个模型变量`net`，它是一个`Sequential`类的实例。`Sequential`类为串联在一起的多个层定义了一个容器。当给定输入数据，`Sequential`实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，依此类推。在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`会让你熟悉标准的流水线。

回顾图3.1.2中的单层网络架构，这一单层被称为*全连接层*（fully-connected layer），因为它的每一个输入都通过矩阵-向量乘法连接到它的每个输出。

在OneFlow中，全连接层在`Linear`类中定义。值得注意的是，我们将两个参数传递到`nn.Linear`中。第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

```python
# `nn` 是神经网络的缩写
from oneflow import nn
net = nn.Sequential(nn.Linear(2, 1))
```

## 3.3.4. 初始化模型参数

在使用`net`之前，我们需要初始化模型参数。如在线性回归模型中的权重和偏置。
深度学习框架通常有预定义的方法来初始化参数。
在这里，我们指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，偏置参数将初始化为零。

正如我们在构造`nn.Linear`时指定输入和输出尺寸一样。现在我们直接访问参数以设定初始值。我们通过`net[0]`选择网络中的第一个图层，然后使用`weight.data`和`bias.data`方法访问参数。然后使用替换方法`normal_`和`fill_`来重写参数值。

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```
    tensor([0.], dtype=oneflow.float32, requires_grad=True)

## 3.3.5. 定义损失函数

计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数。默认情况下，它返回所有样本损失的平均值。

```python
loss = nn.MSELoss()
```

## 3.3.6. 定义优化算法

小批量随机梯度下降算法是一种优化神经网络的标准工具，OneFlow在`optim`模块中实现了该算法的许多变种。当我们实例化`SGD`实例时，我们要指定优化的参数（可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。

```python
trainer = flow.optim.SGD(net.parameters(), lr=0.03)
```

## 3.3.7. 训练

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。
我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。
当我们需要更复杂的模型时，高级API的优势将大大增加。
当我们有了所有的基本组件，训练过程代码与我们从零开始实现时所做的非常相似。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（`train_data`），不停地从中获取一个小批量的输入和相应的标签。对于每一个小批量，我们会进行以下步骤:

* 通过调用`net(X)`生成预测并计算损失`l`（正向传播）。
* 通过进行反向传播来计算梯度。
* 通过调用优化器来更新模型参数。

为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```
    epoch 1, loss 0.000198
    epoch 2, loss 0.000101
    epoch 3, loss 0.000100

下面我们比较生成数据集的真实参数和通过有限数据训练获得的模型参数。
要访问参数，我们首先从`net`访问所需的层，然后读取该层的权重和偏置。
正如在从零开始实现中一样，我们估计得到的参数与生成数据的真实参数非常接近。

```python
w = net[0].weight.data
print('w的估计误差：', true_w - flow.reshape(w, true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```
    w的估计误差： tensor([7.4911e-04, 1.6212e-05], dtype=oneflow.float32,
       grad_fn=<broadcast_sub_backward>)
    b的估计误差： tensor([0.0001], dtype=oneflow.float32, grad_fn=<scalar_add_backward>)

## 3.3.8. 小结

* 我们可以使用OneFlow的高级API更简洁地实现模型。
* 在OneFlow中，`data`模块提供了数据处理工具，`nn`模块定义了大量的神经网络层和常见损失函数。
* 我们可以通过`_`结尾的方法将参数替换，从而初始化参数。


## 3.3.9. 练习

1. 如果我们用`nn.MSELoss()`替换`nn.MSELoss(reduction='sum')`，为了使代码的行为相同，需要怎么更改学习速率？为什么？
2. 查看OneFlow文档，了解提供了哪些损失函数和初始化方法。用SmoothL1Loss损失来代替。
3. 你如何访问`net[0].weight`的梯度？
