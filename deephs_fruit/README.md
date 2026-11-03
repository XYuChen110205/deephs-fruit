# Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning
> 基于高光谱成像与深度学习的水果成熟度测量

    Version 2:
        - The training is now more stable
        - Added new hyperspectral recordings
        - Added HyveConv++ model
> 版本2：
>     - 训练现在更加稳定
>     - 新增了高光谱记录数据
>     - 新增了 HyveConv++ 模型

    Version 1 is still avaiable on branch 'paper_version' 
> 版本1 仍在 'paper_version' 分支上可用

Here you can find the dataset and the official implementation of the HS-CNN network with an optimized training pipeline.
> 在这里你可以找到数据集以及 HS-CNN 网络的官方实现，附带优化的训练流程。

## Data set (v2)
> ## 数据集 (v2)

The dataset is [here](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/) or as [torrent](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/DeepHS-Fruit-2023-Datasets.torrent) available. 
> 数据集可在[此处](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/)获取，或作为[种子文件](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/DeepHS-Fruit-2023-Datasets.torrent)下载。

It contains recordings of:
> 包含以下水果的记录：

 - Avocados
 > - 牛油果
 - Kiwis
 > - 猕猴桃
 - Persimmons
 > - 柿子
 - Papayas
 > - 木瓜
 - Mango
 > - 芒果

Three hyperspectral cameras were use:
> 使用了三台高光谱相机：

 - Specim FX 10
 > - Specim FX 10
 - INNO-SPEC Redeye 1.7
 > - INNO-SPEC Redeye 1.7
 - Corning microHSI 410 Vis-NIR Hyperspectral Sensor
 > - Corning microHSI 410 Vis-NIR 高光谱传感器

The data set covers four measurement series. Labels are provided as destructive measurements (fruit flesh firmness, sugar content and overall ripeness)
> 数据集涵盖四个测量系列。标签以破坏性测量方式提供（果肉硬度、糖度和整体成熟度）

### Requirements
> ### 环境要求

 - Python 3.10
 > - Python 3.10
 - PyTorch 1.11.0
 > - PyTorch 1.11.0

and the packages defined in the requirements file (```pip3 install -r requirements.txt```)
> 以及 requirements 文件中定义的包（```pip3 install -r requirements.txt```）

 - Download the data set to a local folder
 > - 将数据集下载到本地文件夹

## How to train
> ## 如何训练

If all packages are installed and the data set was downloaded, the training can start.
> 如果所有包都已安装且数据集已下载，即可开始训练。

This will train the HS-CNN model on the ripeness classification of avocados:
> 以下命令将训练 HS-CNN 模型用于牛油果成熟度分类：

    PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --data_path /folder/of/downloaded/dataset/ --model deephs_net --fruit avocado --classification_type ripeness --seed 23312323
>     PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --data_path /下载数据集的路径/ --model deephs_net --fruit avocado --classification_type ripeness --seed 23312323

<img src="images/deephs_net_loss.png" alt="Loss" style="width: 300px;"/><br>
> <img src="images/deephs_net_loss.png" alt="损失" style="width: 300px;"/><br>

<img src="images/deephs_net_accuracy.png" alt="Accuracy" style="width: 300px;"/><br>
> <img src="images/deephs_net_accuracy.png" alt="准确率" style="width: 300px;"/><br>

<img src="images/deephs_net_confusion.png" alt="Confusion" style="width: 300px;"/><br>
> <img src="images/deephs_net_confusion.png" alt="混淆矩阵" style="width: 300px;"/><br>

**Figure 1** - Training of HS-CNN:
> **图1** - HS-CNN 的训练过程：
  
And this will train HS-CNN + HyveConv++ on the same classification task:
> 以下命令将在相同分类任务上训练 HS-CNN + HyveConv++：

    PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --data_path /folder/of/downloaded/dataset/ --model hyve --fruit avocado --classification_type ripeness --seed 23312323
>     PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --data_path /下载数据集的路径/ --model hyve --fruit avocado --classification_type ripeness --seed 23312323

<img src="images/hyve_loss.png" alt="Loss" style="width: 300px;"/><br>
> <img src="images/hyve_loss.png" alt="损失" style="width: 300px;"/><br>

<img src="images/hyve_accuracy.png" alt="Accuracy" style="width: 300px;"/><br>
> <img src="images/hyve_accuracy.png" alt="准确率" style="width: 300px;"/><br>

<img src="images/hyve_confusion.png" alt="Confusion" style="width: 300px;"/><br>
> <img src="images/hyve_confusion.png" alt="混淆矩阵" style="width: 300px;"/><br>

**Figure 2** - Training of HS-CNN + HyveConv++:
> **图2** - HS-CNN + HyveConv++ 的训练过程：

```PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --help``` provides helpful information regarding the parameters. 
> ```PYTHONPATH=$PYTHONPATH:. python3 classification/train.py --help``` 提供有关参数的有用信息。

For more information about the training framework PyTorch-Lightning, we refer to the official documentation (https://pytorch-lightning.readthedocs.io/en/latest/).
> 有关训练框架 PyTorch-Lightning 的更多信息，请参阅官方文档 (https://pytorch-lightning.readthedocs.io/en/latest/)。

            
## Citation
> ## 引用

The paper was presented on IJCNN 2021.
> 该论文在 IJCNN 2021 上发表。

```
@inproceedings{Varga2021,
abstract = {We present a system to measure the ripeness of fruit with a hyperspectral camera and a suitable deep neural network architecture. This architecture did outperform competitive baseline models on the prediction of the ripeness state of fruit. For this, we recorded a data set of ripening avocados and kiwis, which we make public. We also describe the process of data collection in a manner that the adaption for other fruit is easy. The trained network is validated empirically, and we investigate the trained features. Furthermore, a technique is introduced to visualize the ripening process.},
archivePrefix = {arXiv},
arxivId = {2104.09808},
author = {Varga, Leon Amadeus and Makowski, Jan and Zell, Andreas},
booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
doi = {10.1109/IJCNN52387.2021.9533728},
eprint = {2104.09808},
isbn = {978-1-6654-3900-8},
keywords = {Index Terms-hyperspectral,convolutional neu-ral network,deep learning,ripening fruit},
month = {jul},
pages = {1--8},
publisher = {IEEE},
title = {{Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning}},
url = {https://arxiv.org/abs/2104.09808v1 http://arxiv.org/abs/2104.09808 https://ieeexplore.ieee.org/document/9533728/},
year = {2021}
}
```
> ```
> @inproceedings{Varga2021,
> abstract = {我们提出了一种利用高光谱相机和合适的深度神经网络架构来测量水果成熟度的系统。该架构在预测水果成熟度方面优于竞争基线模型。为此，我们记录了牛油果和猕猴桃的成熟数据集，并将其公开。我们还描述了数据收集过程，以便其他水果的适配变得容易。训练后的网络经过实证验证，我们研究了训练得到的特征。此外，还介绍了一种可视化成熟过程的技术。},
> archivePrefix = {arXiv},
> arxivId = {2104.09808},
> author = {Varga, Leon Amadeus and Makowski, Jan and Zell, Andreas},
> booktitle = {2021 International Joint Conference on Neural Networks (IJCNN)},
> doi = {10.1109/IJCNN52387.2021.9533728},
> eprint = {2104.09808},
> isbn = {978-1-6654-3900-8},
> keywords = {关键词-高光谱,卷积神经网络,深度学习,成熟水果},
> month = {jul},
> pages = {1--8},
> publisher = {IEEE},
> title = {{基于高光谱成像与深度学习的水果成熟度测量}},
> url = {https://arxiv.org/abs/2104.09808v1 http://arxiv.org/abs/2104.09808 https://ieeexplore.ieee.org/document/9533728/},
> year = {2021}
> }
> ```

For HyveConv++ please check: https://github.com/cogsys-tuebingen/hyve_conv
> 有关 HyveConv++ 的更多信息，请参阅：https://github.com/cogsys-tuebingen/hyve_conv
