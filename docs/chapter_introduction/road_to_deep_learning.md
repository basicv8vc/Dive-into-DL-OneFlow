# 深度学习之路

大约2010年开始，那些在计算上看起来不可行的神经网络算法变得热门起来，实际上是以下两点导致的。
其一，随着互联网的公司的出现，为数亿在线用户提供服务，大规模数据集变得触手可及。
另外，廉价又高质量的传感器、廉价的数据存储（克里德定律）以及廉价计算（摩尔定律）的普及，特别是GPU的普及，使大规模算力唾手可得。这一点在表1.5.1中得到了说明。


<left>表1.5.1 数据集vs计算机内存和计算能力</left>

|年代|数据规模|内存|每秒浮点运算|
|:--|:-|:-|:-|
|1970|100 （虹膜）|1 KB|100 KF (Intel 8080)|
|1980|1 K （波士顿房价）|100 KB|1 MF (Intel 80186)|
|1990|10 K （光学字符识别）|10 MB|10 MF (Intel 80486)|
|2000|10 M （网页）|100 MB|1 GF (Intel Core)|
|2010|10 G （广告）|1 GB|1 TF (Nvidia C2050)|
|2020|1 T （社交网络）|100 GB|1 PF (Nvidia DGX-2)|


很明显，随机存取存储器没有跟上数据增长的步伐。
与此同时，算力的增长速度已经超过了现有数据的增长速度。
这意味着统计模型需要提高内存效率（这通常是通过添加非线性来实现的），同时由于计算预算的增加，能够花费更多时间来优化这些参数。
因此，机器学习和统计的关注点从（广义的）线性模型和核方法转移到了深度神经网络。
这也是为什么许多深度学习的中流砥柱，如多层感知机[3]、卷积神经网络[4]、长短期记忆网络[5]和Q学习[6]，在相当长一段时间处于相对休眠状态之后，在过去十年中被“重新发现”的原因之一。

最近十年，在统计模型、应用和算法方面的进展就像寒武纪大爆发。
事实上，最先进的技术不仅仅是应用于几十年前的算法的可用资源的结果。
下面列举了帮助研究人员在过去十年中取得巨大进步的想法（虽然只是触及了的皮毛）。


* 新的容量控制方法，如*dropout*[7]，有助于减轻过拟合的危险。这是通过在整个神经网络中应用噪声注入[8]来实现的，出于训练目的，用随机变量来代替权重。
* 注意力机制解决了困扰统计学一个多世纪的问题：如何在不增加可学习参数的情况下增加系统的记忆和复杂性。研究人员通过使用只能被视为可学习的指针结构[9]找到了一个优雅的解决方案。不需要记住整个文本序列(例如用于固定维度表示中的机器翻译)，所有需要存储的都是指向翻译过程的中间状态的指针。这大大提高了长序列的准确性，因为模型在开始生成新序列之前不再需要记住整个序列。
* 多阶段设计。例如，存储器网络[10]和神经编程器-解释器[11]。它们允许统计建模者描述用于推理的迭代方法。这些工具允许重复修改深度神经网络的内部状态，从而执行推理链中的后续步骤，类似于处理器如何修改用于计算的存储器。
* 另一个关键的发展是生成对抗网络[12]的发明。传统模型中，密度估计和生成模型的统计方法侧重于找到合适的概率分布和（通常是近似的）抽样算法。因此，这些算法在很大程度上受到统计模型固有灵活性的限制。生成式对抗性网络的关键创新是用具有可微参数的任意算法代替采样器。然后对这些数据进行调整，使得鉴别器（实际上是对两个样本的测试）不能区分假数据和真实数据。通过使用任意算法生成数据的能力，它为各种技术打开了密度估计的大门。驰骋的斑马[13]和假名人脸[14]的例子都证明了这一进展。即使是业余的涂鸦者也可以根据描述场景布局的草图生成照片级真实图像[15]。
* 在许多情况下，单个GPU不足以处理可用于训练的大量数据。在过去的十年中，构建并行和分布式训练算法的能力有了显著提高。设计可伸缩算法的关键挑战之一是深度学习优化的主力——随机梯度下降，它依赖于相对较小的小批量数据来处理。同时，小批量限制了GPU的效率。因此，在1024个GPU上进行训练，例如每批32个图像的小批量大小相当于总计约32000个图像的小批量。最近的工作，首先是由[16]完成的，随后是[17]和[18]，将观察大小提高到64000个，将ResNet-50模型在ImageNet数据集上的训练时间减少到不到7分钟。作为比较——最初的训练时间是按天为单位的。
* 并行计算的能力也对强化学习的进步做出了相当关键的贡献。这导致了计算机在围棋、雅达里游戏、星际争霸和物理模拟（例如，使用MuJoCo）中实现超人性能的重大进步。有关如何在AlphaGo中实现这一点的说明，请参见如[19]。简而言之，如果有大量的（状态、动作、奖励）三元组可用，即只要有可能尝试很多东西来了解它们之间的关系，强化学习就会发挥最好的作用。仿真提供了这样一条途径。
* 深度学习框架在传播思想方面发挥了至关重要的作用。允许轻松建模的第一代框架包括[Caffe](https://github.com/BVLC/caffe)、[Torch](https://github.com/torch)和[Theano](https://github.com/Theano/Theano)。许多开创性的论文都是用这些工具写的。到目前为止，它们已经被[TensorFlow](https://github.com/tensorflow/tensorflow)（通常通过其高级API [Keras](https://github.com/keras-team/keras)使用）、[CNTK](https://github.com/Microsoft/CNTK)、[Caffe 2](https://github.com/caffe2/caffe2)和[Apache MXNet](https://github.com/apache/incubator-mxnet)所取代。第三代工具，即用于深度学习的命令式工具，可以说是由[Chainer](https://github.com/chainer/chainer)率先推出的，它使用类似于Python NumPy的语法来描述模型。这个想法被[PyTorch](https://github.com/pytorch/pytorch)、MXNet的[Gluon API](https://github.com/apache/incubator-mxnet)和[Jax](https://github.com/google/jax)都采纳了。

“系统研究人员构建更好的工具”和“统计建模人员构建更好的神经网络”之间的分工大大简化了事情。
例如，在2014年，对于卡内基梅隆大学机器学习博士生来说，训练线性回归模型曾经是一个不容易的作业问题。
而现在，这项任务只需不到10行代码就能完成，这让每个程序员轻易掌握了它。
