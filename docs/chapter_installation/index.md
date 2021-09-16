# 安装

我们需要配置一个环境来运行 Python、Jupyter Notebook、相关库以及运行本书所需的代码，以快速入门并获得动手学习经验。

## 安装 Miniconda

最简单的方法就是安装依赖 Python 3.x 的 [Miniconda](https://conda.io/en/latest/miniconda.html)。如果已安装 conda，则可以跳过以下步骤。
从网站下载相应的 Miniconda sh 文件，然后使用 `sh <FILENAME> -b` 从命令行执行安装。

对于 macOS 用户：

```bash
# 文件名可能会更改
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

对于 Linux 用户：

```bash
# 文件名可能会更改
sh Miniconda3-latest-Linux-x86_64.sh -b
```


接下来，初始化终端 Shell，以便我们可以直接运行 `conda`。

```bash
~/miniconda3/bin/conda init
```

## 创建新的虚拟环境

建议使用一个新的Python虚拟环境来执行本项目的代码，虚拟环境的好处这里就不多说了。现在关闭并重新打开当前的 shell。你应该能用下面的命令创建一个新的环境：

```bash
conda create --name d2l python=3.7 -y
```


现在我们要激活 `d2l` 环境。

```bash
conda activate d2l
```

## 下载 Dive-into-DL-OneFlow项目

接下来，需要下载这本书的代码：

```bash
git clone https://github.com/basicv8vc/Dive-into-DL-OneFlow
```


## 安装OneFlow框架


在安装深度学习框架之前，请先检查你的计算机上是否有可用的 GPU（在笔记本电脑上为显示器提供输出的GPU不算）。

**注意这里一律安装nightly版OneFlow**

安装GPU版OneFlow

```bash
python -m pip install -f https://release.oneflow.info oneflow==0.5rc1+cu102
```
当然也可以安装CPU版OneFlow或其他CUDA版本：

```
python -m pip install --find-links https://release.oneflow.info oneflow==0.5rc1+[PLATFORM]
```

合理的`[PLATFORM]`字段如下表中第一列:

| PLATFORM |CUDA Driver Version| Supported GPUs |
|---|---|---|
| cu112  | >= 450.80.02  | GTX 10xx, RTX 20xx, A100, RTX 30xx |
| cu111  | >= 450.80.02  | GTX 10xx, RTX 20xx, A100, RTX 30xx |
| cu110, cu110_xla  | >= 450.36.06  | GTX 10xx, RTX 20xx, A100|
| cu102, cu102_xla  | >= 440.33  | GTX 10xx, RTX 20xx |
| cu101, cu101_xla  | >= 418.39  | GTX 10xx, RTX 20xx |
| cu100, cu100_xla  | >= 410.48  | GTX 10xx, RTX 20xx |
| cpu  | N/A | N/A |

对于国内的小伙伴，可以选择[清华源](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)：

```
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```


安装完成后，我们通过运行以下命令打开 Jupyter 笔记本：

```bash
jupyter notebook
```

现在，你可以在 Web 浏览器中打开 <http://localhost:8888>（通常会自动打开）。然后我们可以运行这本书中每个部分的代码。在运行书籍代码、更新OneFlow框架之前，请始终执行 `conda activate d2l` 以激活运行时环境。要退出环境，请运行 `conda deactivate`。

