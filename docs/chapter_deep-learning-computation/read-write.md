# 读写文件

到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们对所学的模型足够满意，我们希望保存训练的模型以备将来在各种环境中使用（甚至可能在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果（检查点），以确保在服务器电源被不小心断掉时不会损失几天的计算结果。因此，现在是时候学习如何加载和存储权重向量和整个模型。本节将讨论这些问题。

## 5.5.1. 加载和保存张量

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。注意`save`只能存储字典类型的数据。当我们要读取或写入模型中的所有权重时，这很方便。

```python
import oneflow as flow
from oneflow import nn
import oneflow.nn.functional as F

x = flow.arange(4)
flow.save({'x': x}, 'x-file')
```

我们现在可以将存储在文件中的数据读回内存。

```python
x2 = flow.load('x-file')
x2
```
    {'x': tensor([0, 1, 2, 3], dtype=oneflow.int64)}

再来看一个例子，

```python
y = flow.zeros(4)
params = dict({'x': x, 'y':y})
flow.save(params,'x-files')
params = flow.load('x-files')
params
```
    {'y': tensor([0., 0., 0., 0.], dtype=oneflow.float32),
     'x': tensor([0, 1, 2, 3], dtype=oneflow.int64)}


## 5.5.2. 加载和保存模型参数

保存单个权重向量（或其他张量）确实是有用的，但是如果我们想保存整个模型，并在以后加载它们，单独保存每个向量则会变得很麻烦。毕竟，我们可能有数百个参数散布在各处。因此，深度学习框架提供了内置函数来保存和加载整个网络。需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。例如，如果我们有一个3层多层感知机，我们需要单独指定结构。因为模型本身可以包含任意代码，所以模型本身难以序列化。因此，为了恢复模型，我们需要用代码生成结构，然后从磁盘加载参数。让我们从熟悉的多层感知机开始尝试一下。

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = flow.randn(2, 20)
Y = net(X)
```

接下来，我们将模型的参数存储为一个叫做“mlp.params”的文件。

```python
flow.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，我们实例化了原始多层感知机模型的一个备份。我们没有随机初始化模型参数，而是直接读取文件中存储的参数。

```python
clone = MLP()
clone.load_state_dict(flow.load('mlp.params'))
clone.eval()
```
    MLP(
     (hidden): Linear(in_features=20, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
    )

由于两个实例具有相同的模型参数，在输入相同的`X`时，两个实例的计算结果应该相同。让我们来验证一下。

```python
Y_clone = clone(X)
Y_clone == Y
```
    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=oneflow.int8)

## 5.5.3. 小结

* `save`和`load`函数可用于张量对象的文件读写。
* 我们可以通过参数字典保存和加载网络的全部参数。
* 保存结构必须在代码中完成，而不是在参数中完成。

## 5.5.4. 练习

1. 即使不需要将经过训练的模型部署到不同的设备上，存储模型参数还有什么实际的好处？
1. 假设我们只想复用网络的一部分，以将其合并到不同的网络结构中。比如说，如果你想在一个新的网络中使用之前网络的前两层，你该怎么做？
1. 如何同时保存网络结构和参数？你会对结构加上什么限制？
