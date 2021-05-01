### 项目介绍

通过引入测地距离改进MCFS算法，将IsoMap与MCFS相结合的特征选择算法MCFS-I，算法示例在./src/test.py文件中。

### 文件介绍

- data文件夹中包含本次实验使用的数据集：MNIST、Yale、warpPIE10P、lung_small，digits数据集则直接调用sklearn包中的数据集。
- data_info.txt文件中包含本次实验所有数据集的详细信息。
- cmp文件夹中包含本次实验的对比算法和一些工具。
- MCFS文件夹中包含MCFS算法和MCFS-I算法。
  - mcfs中的参数i用于选择算法，当i=0时使用MCFS，当i=1时使用MCFS-I
- Result文件夹中为本次实验产生的结果。
- Result_excel为用Result文件夹中的结果生成的excel，用于使用hiplot画图。
- pic文件夹中为使用hiplot进行画的图，也可使用matplotlib画图。
- src为实验的源代码，utils为测试时使用的工具，config为配置文件
  - test用于测试每个算法对于每个数据集的特征选择后使用K-means后的NMI
  - test_neighbor用于测试不同参数n_neighbor对聚类结果的影响
  - test_nemb用于测试不同参数n_emb对于聚类结果的影响

### 运行环境

- numpy==1.19.2
- python==3.6.13
- pytorch==1.6.0
- scikit-learn==0.24.1