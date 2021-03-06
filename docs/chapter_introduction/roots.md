

# 起源

为解决各式机器学习问题，深度学习提供了强大的工具。
虽然许多深度学习方法都是最近的才有重大突破，但使用数据和神经网络编程的核心思想已经研究了几个世纪。
事实上，人类长期以来就有分析数据和预测未来结果的愿望，而自然科学的大部分都植根于此。
例如，伯努利分布是以[雅各布•贝努利（1655--1705）](https://en.wikipedia.org/wiki/Jacob\uBernoulli)命名的。
而高斯分布是由[卡尔•弗里德里希•高斯（1777—1855）](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)发现的，
他发明了最小均方算法，至今仍用于解决从保险计算到医疗诊断的许多问题。
这些工具算法在自然科学中产生了一种实验方法——例如，电阻中电流和电压的欧姆定律可以用线性模型完美地描述。

即使在中世纪，数学家对*估计*（estimation）也有敏锐的直觉。
例如，[雅各布·克贝尔 (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)的几何学书籍举例说明，通过平均16名成年男性的脚的长度，可以得出一英尺的长度。

<div align=center>
<img width=500 src="../img/koebel.jpg"/>
</div>
<center>图1.4.1 估计一英尺的长度。</center>

图1.4.1说明了这个估计器是如何工作的。
16名成年男子被要求脚连脚排成一行。
然后将它们的总长度除以16，得到现在等于1英尺的估计值。
这个算法后来被改进以处理畸形的脚——将拥有最短和最长脚的两个人送走，对其余的人取平均值。
这是最早的修剪均值估计的例子之一。

随着数据的收集和可获得性，统计数据真正实现了腾飞。
[罗纳德·费舍尔（1890-1962）](https://en.wikipedia.org/wiki/Ronald_-Fisher)对统计理论和在遗传学中的应用做出了重大贡献。
他的许多算法（如线性判别分析）和公式（如费舍尔信息矩阵）至今仍被频繁使用。
甚至，费舍尔在1936年发布的虹膜数据集，有时仍然被用来解读机器学习算法。
他也是优生学的倡导者，这提醒我们：使用数据科学虽然在道德上存在疑问，但是与数据科学在工业和自然科学中的生产性使用一样，有着悠久的历史。

机器学习的第二个影响来自[克劳德·香农(1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon)的信息论和[艾伦·图灵（1912-1954）](https://en.wikipedia.org/wiki/Alan_Turing)的计算理论。
图灵在他著名的论文《计算机器与智能》[1]中提出了“机器能思考吗？”的问题。
在他所描述的图灵测试中，如果人类评估者很难根据文本互动区分机器和人类的回答，那么机器就可以被认为是“智能的”。

另一个影响可以在神经科学和心理学中找到。
其中，最古老的算法之一是[唐纳德·赫布 (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb)开创性的著作《行为的组织》[2]。
他提出神经元通过积极强化学习，是Rosenblatt感知器学习算法的原型，被称为“赫布学习”。
这个算法也为当今深度学习的许多随机梯度下降算法奠定了基础：强化期望行为和减少不良行为，以获得神经网络中参数的良好设置。

*神经网络*（neural networks）得名的原因是生物灵感。
一个多世纪以来（追溯到1873年亚历山大·贝恩和1890年詹姆斯·谢林顿的模型），研究人员一直试图组装类似于相互作用的神经元网络的计算电路。
随着时间的推移，对生物学的解释变得不再肤浅，但这个名字仍然存在。
其核心是当今大多数网络中都可以找到的几个关键原则：

* 线性和非线性处理单元的交替，通常称为*层*（layers）。
* 使用链式规则（也称为*反向传播*（backpropagation））一次性调整网络中的全部参数。

在最初的快速发展之后，神经网络的研究从1995年左右一直开始停滞不前，直到到2005年才稍有起色。
这主要是因为两个原因。
首先，训练网络（在计算上）非常昂贵。在
上个世纪末，随机存取存储器（RAM）非常强大，而计算能力却很弱。
其次，数据集相对较小。
事实上，费舍尔1932年的虹膜数据集是测试算法有效性的流行工具，
而MNIST数据集的60000个手写数字的数据集被认为是巨大的。
考虑到数据和计算的稀缺性，*核方法*（kernel method）、*决策树*（decision tree）和*图模型*（graph models）等强大的统计工具（在经验上）证明是更为优越的。
与神经网络不同的是，这些算法不需要数周的训练，而且有很强的理论依据，可以提供可预测的结果。