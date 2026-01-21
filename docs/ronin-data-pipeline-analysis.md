# Ronin数据管道分析

## 数据加载器位置
- `external/ronin/source/data_glob_speed.py`: DenseSequenceDataset, StridedSequenceDataset, SequenceToSequenceDataset
- `external/ronin/source/data_glob_heading.py`: HeadingDataset
- `external/ronin/source/baselines/data_stabilized_local_speed.py`: StabilizedLocalSpeedDataset
- `external/ronin/source/data_utils.py`: 数据工具函数和CompiledSequence类

## 数据格式
- 数据存储在HDF5文件中（.h5格式）
- 每个序列包含IMU数据（加速度计和陀螺仪）和地面真实速度
- 数据预处理包括窗口化和标准化

## 预处理步骤
1. 从HDF5文件加载原始IMU数据
2. 应用窗口化和步长处理
3. 数据标准化（减去均值，除以标准差）
4. 转换为PyTorch张量

## 数据集划分
- `external/ronin/lists/list_train.txt`: 训练集序列列表
- `external/ronin/lists/list_val.txt`: 验证集序列列表
- `external/ronin/lists/list_test_seen.txt`: 测试集（已见场景）
- `external/ronin/lists/list_test_unseen.txt`: 测试集（未见场景）

## 需要复制的文件
1. `external/ronin/source/data_utils.py` - 核心数据工具函数
2. `external/ronin/source/data_glob_speed.py` - 速度数据集类
3. `external/ronin/lists/` 目录 - 数据集划分文件
4. `external/ronin/source/math_util.py` - 数学工具函数