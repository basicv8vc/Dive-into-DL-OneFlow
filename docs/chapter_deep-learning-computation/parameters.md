# 参数管理

一旦我们选择了架构并设置了超参数，我们就进入了训练阶段。此时，我们的目标是找到使损失函数最小化的参数值。经过训练后，我们将需要使用这些参数来做出未来的预测。此外，有时我们希望提取参数，以便在其他环境中复用它们，将模型保存到磁盘，以便它可以在其他软件中执行，或者为了获得科学的理解而进行检查。

大多数情况下，我们可以忽略声明和操作参数的具体细节，而只依靠深度学习框架来完成繁重的工作。然而，当我们离开具有标准层的层叠架构时，我们有时会陷入声明和操作参数的麻烦中。在本节中，我们将介绍以下内容：

* 访问参数，用于调试、诊断和可视化。
* 参数初始化。
* 在不同模型组件间共享参数。

我们首先关注具有单隐藏层的多层感知机。

```python
import oneflow as flow
from oneflow import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = flow.rand(2, 4)
net(X)
```
    tensor([[0.2993],
        [0.1468]], dtype=oneflow.float32, grad_fn=<broadcast_add_backward>)

## 5.2.1. 参数访问

我们从已有模型中访问参数。当通过`Sequential`类定义模型时，我们可以通过索引来访问模型的任意层。这就像模型是一个列表一样。每层的参数都在其属性中。如下所示，我们可以检查第二个全连接层的参数。

```python
print(net[2].state_dict())
```
    OrderedDict([('weight', tensor([[-0.3209,  0.3443,  0.2851, -0.2124,  0.1451, -0.3104,  0.2327,  0.0721]],
       dtype=oneflow.float32, grad_fn=<accumulate_grad>)), ('bias', tensor([-0.2249], dtype=oneflow.float32, grad_fn=<accumulate_grad>))])

输出的结果告诉我们一些重要的事情。首先，这个全连接层包含两个参数，分别是该层的权重和偏置。两者都存储为单精度浮点数（float32）。注意，参数名称允许我们唯一地标识每个参数，即使在包含数百个层的网络中也是如此。

### 5.2.1.1. 目标参数

注意，每个参数都表示为参数（parameter）类的一个实例。要对参数执行任何操作，首先我们需要访问底层的数值。有几种方法可以做到这一点。有些比较简单，而另一些则比较通用。下面的代码从第二个神经网络层提取偏置，提取后返回的是一个参数类实例，并进一步访问该参数的值。

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```
    <class 'oneflow._oneflow_internal.nn.Parameter'>
    tensor([-0.2249], dtype=oneflow.float32, grad_fn=<accumulate_grad>)
    tensor([-0.2249], dtype=oneflow.float32, requires_grad=True)

参数是复合的对象，包含值、梯度和额外信息。这就是我们需要显式请求值的原因。

除了值之外，我们还可以访问每个参数的梯度。由于我们还没有调用这个网络的反向传播，所以参数的梯度处于初始状态。

```python
net[2].weight.grad == None
```
    True

### 5.2.1.2. 一次性访问所有参数

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂，因为我们需要递归整个树来提取每个子块的参数。下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```
    ('weight', flow.Size([8, 4])) ('bias', flow.Size([8]))
    ('0.weight', flow.Size([8, 4])) ('0.bias', flow.Size([8])) ('2.weight', flow.Size([1, 8])) ('2.bias', flow.Size([1]))

这为我们提供了另一种访问网络参数的方式，如下所示。

```python
net.state_dict()['2.bias'].data
```
    tensor([-0.2249], dtype=oneflow.float32, requires_grad=True)

### 5.2.1.3. 从嵌套块收集参数

让我们看看，如果我们将多个块相互嵌套，参数命名约定是如何工作的。为此，我们首先定义一个生成块的函数（可以说是块工厂），然后将这些块组合到更大的块中。

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```
    tensor([[0.4148],
        [0.4149]], dtype=oneflow.float32, grad_fn=<broadcast_add_backward>)

现在我们已经设计了网络，让我们看看它是如何组织的。

```python
print(rgnet)
```
    Sequential(
    (0): Sequential(
        (block 0): Sequential(
        (0): Linear(in_features=4, out_features=8, bias=True)
        (1): ReLU()
        (2): Linear(in_features=8, out_features=4, bias=True)
        (3): ReLU()
        )
        (block 1): Sequential(
        (0): Linear(in_features=4, out_features=8, bias=True)
        (1): ReLU()
        (2): Linear(in_features=8, out_features=4, bias=True)
        (3): ReLU()
        )
        (block 2): Sequential(
        (0): Linear(in_features=4, out_features=8, bias=True)
        (1): ReLU()
        (2): Linear(in_features=8, out_features=4, bias=True)
        (3): ReLU()
        )
        (block 3): Sequential(
        (0): Linear(in_features=4, out_features=8, bias=True)
        (1): ReLU()
        (2): Linear(in_features=8, out_features=4, bias=True)
        (3): ReLU()
        )
    )
    (1): Linear(in_features=4, out_features=1, bias=True)
    )

因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。例如，我们下面访问第一个主要的块，其中第二个子块的第一层的偏置项。

```python
rgnet[0][1][0].bias.data
```
    tensor([ 0.2862,  0.2679, -0.0803, -0.4119, -0.1303,  0.2883, -0.2981, -0.2536],
       dtype=oneflow.float32, requires_grad=True)

## 5.2.2. 参数初始化

我们知道了如何访问参数，现在让我们看看如何正确地初始化参数。我们在4.8节中讨论了良好初始化的必要性。深度学习框架提供默认随机初始化。然而，我们经常希望根据其他规则初始化权重。深度学习框架提供了最常用的规则，也允许创建自定义初始化方法。

默认情况下，OneFlow会根据一个范围均匀地初始化权重和偏置矩阵，这个范围是根据输入和输出维度计算出的。OneFlow的`nn.init`模块提供了多种预置初始化方法。

### 5.2.2.1. 内置初始化

让我们首先调用内置的初始化器。下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，且将偏置参数设置为0。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```
    tensor([-0.0226, -0.0055,  0.0055, -0.0047], dtype=oneflow.float32,
        grad_fn=<reshape_backward>),
    tensor(0., dtype=oneflow.float32, grad_fn=<slice_backward>)

我们还可以将所有参数初始化为给定的常数（比如1）。

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```
    tensor([1., 1., 1., 1.], dtype=oneflow.float32, grad_fn=<reshape_backward>),
    tensor(0., dtype=oneflow.float32, grad_fn=<slice_backward>)

我们还可以对某些块应用不同的初始化方法。例如，下面我们使用Xavier初始化方法初始化第一层，然后第二层初始化为常量值42。

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```
    tensor([-0.1513, -0.6396,  0.1588,  0.6312], dtype=oneflow.float32,
       grad_fn=<reshape_backward>)
    tensor([[42., 42., 42., 42., 42., 42., 42., 42.]], dtype=oneflow.float32,
       requires_grad=True)

### 5.2.2.2. 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
\tag{5.2.1}
$$

同样，我们实现了一个`my_init`函数来应用到`net`。

```python
def my_init(m):
    with flow.no_grad():
        if type(m) == nn.Linear:
            print("Init", *[(name, param.shape)
                            for name, param in m.named_parameters()][0])
            nn.init.uniform_(m.weight, -10, 10)
            factor = (m.weight.data.abs() >= 5).type_as(m.weight.data)
            m.weight.data[:] *= factor

net.apply(my_init)
net[0].weight[:2]
```
    Init weight flow.Size([8, 4])
    Init weight flow.Size([1, 8])
    tensor([[-0.0000, -8.8609, -9.5075, -5.6164],
            [ 0.0000,  0.0000,  9.7500,  9.6698]], dtype=oneflow.float32,
        grad_fn=<slice_backward>)

注意，我们始终可以直接设置参数。

```python
with flow.no_grad():
    net[0].weight.data[:] += 1
    net[0].weight.data[0, 0] = 42
    net[0].weight.data[0]
```

## 5.2.3. 参数绑定

有时我们希望在多个层间共享参数。让我们看看如何优雅地做这件事。在下面，我们定义一个稠密层，然后使用它的参数来设置另一个层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
with flow.no_grad():
    net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值。
print(net[2].weight.data[0] == net[4].weight.data[0])
```
    tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=oneflow.int8)
    tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=oneflow.int8)

这个例子表明第二层和第三层的参数是绑定的。它们不仅值相等，而且由相同的张量表示。因此，如果我们改变其中一个参数，另一个参数也会改变。你可能会想，当参数绑定时，梯度会发生什么情况？答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层和第三个隐藏层的梯度会加在一起。

## 5.2.4. 小结

* 我们有几种方法可以访问、初始化和绑定模型参数。
* 我们可以使用自定义初始化方法。

## 5.2.5. 练习

1. 使用5.1节中定义的`FancyMLP`模型，访问各个层的参数。
1. 查看初始化模块文档以了解不同的初始化方法。
1. 构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。
1. 为什么共享参数是个好主意？
