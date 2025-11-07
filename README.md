# 超图权重预测评测框架

用于评测量子纠错中不同超图权重预测方法性能的框架。

**文档导航**:
- [快速开始](QUICKSTART.md) - 5分钟快速上手
- [架构说明](ARCHITECTURE.md) - 深入了解框架设计
- [使用示例](example.py) - 代码示例

## 功能概述

本框架实现了三种超图权重预测方法，并提供了两类评测指标：

### 预测方法

1. **noise_calibration**: 基于噪声校准的方法，通过实际探测器采样数据计算高阶相关性来校准超边概率
2. **correlation**: 基于相关性分析的方法，支持数值方法（高阶相关性）和解析方法（二阶相关性，适用于重复码等简单拓扑）
3. **rl_based**: 基于强化学习（PPO）的方法，通过优化decoder priors来最小化逻辑错误率

### 评测指标

1. **distribution_distance**: 预测概率分布与真实概率分布之间的距离（MAE, RMSE, KL散度等）
2. **decoder_ler**: 不同解码器（PyMatching, PyMatching with correlations等）下的逻辑错误率

## 目录结构

```
stim_predict/
├── README.md                    # 本文档
├── methods/                     # 预测方法实现
│   ├── base.py                  # 基类定义
│   ├── noise_calibration.py    # 噪声校准方法
│   ├── correlation.py           # 相关性方法
│   └── rl_based.py              # 强化学习方法
├── circuits/                    # 电路生成
│   └── circuit_factory.py       # 电路工厂（支持surface code, repetition code等）
├── evaluators/                  # 评测器
│   ├── base.py                  # 基类定义
│   ├── distribution_distance.py # 概率分布距离评测
│   └── decoder_ler.py           # 解码器LER评测
├── utils/                       # 工具函数
│   └── data_manager.py          # 数据管理（保存/加载训练数据和结果）
├── data/                        # 数据存储目录
├── train.py                     # 训练脚本
├── evaluate.py                  # 评测脚本
├── visualize.py                 # 结果可视化工具
├── example.py                   # 使用示例
└── requirements.txt             # 依赖列表
```

## 快速开始

### 1. 简单示例

```python
from circuits import CircuitFactory
from methods import NoiseCalibrationPredictor, CorrelationPredictor
from evaluators import DistributionDistanceEvaluator

# 创建电路
circuit = CircuitFactory.create_circuit(
    code_type='surface_code',
    distance=3,
    rounds=3,
    noise_level=0.001
)

# 采样数据
sampler = circuit.compile_detector_sampler()
detector_samples, observables = sampler.sample(shots=10000, separate_observables=True)

# 训练预测器
predictor = NoiseCalibrationPredictor(num_workers=4)
result = predictor.train(circuit, detector_samples)

# 评测
evaluator = DistributionDistanceEvaluator()
# ... (详见 example.py)
```

### 2. 使用命令行工具

#### 训练

```bash
cd stim_predict
python train.py --experiment my_exp --code-type surface_code --distance 5 --rounds 5 --noise 0.001 --shots 100000 --methods noise_calibration correlation
```

参数说明：
- `--experiment`: 实验名称（必需）
- `--code-type`: 编码类型（surface_code, repetition_code, color_code）
- `--distance`: 码距
- `--rounds`: 测量轮数
- `--noise`: 噪声水平
- `--shots`: 采样次数
- `--methods`: 要训练的方法列表
- `--workers`: 并行工作线程数
- `--rl-epochs`: RL方法的训练轮数（仅用于rl_based方法）
- `--rl-batch-size`: RL方法的批次大小（仅用于rl_based方法）

#### 评测

```bash
python evaluate.py --experiment my_exp --methods noise_calibration correlation --evaluators distribution_distance decoder_ler
```

参数说明：
- `--experiment`: 实验名称（必需）
- `--methods`: 要评测的方法列表
- `--evaluators`: 要使用的评测器（distribution_distance, decoder_ler）
- `--ground-truth`: 真实值来源（默认为'dem'）
- `--decoders`: 要测试的解码器列表（用于decoder_ler评测器）

#### 可视化结果

```bash
python visualize.py --experiment my_exp           # 查看实验结果
python visualize.py --list                        # 列出所有实验
```

### 3. 运行示例

```bash
cd stim_predict
python example.py          # 运行简单示例
python example.py full     # 运行完整流程示例
```

## 使用流程

### 训练阶段

1. 创建量子纠错码电路（使用`CircuitFactory`）
2. 采样探测器数据
3. 训练各个预测方法
4. 保存训练数据和预测结果（使用`DataManager`）

### 评测阶段

1. 加载训练数据和预测结果
2. 获取真实超边概率（从DEM或其他方法）
3. 运行评测器
4. 保存并分析评测结果

## 扩展框架

### 添加新的预测方法

1. 在`methods/`目录下创建新文件
2. 继承`BasePredictor`类
3. 实现`train()`和`predict()`方法
4. 在`methods/__init__.py`中导入

示例：
```python
from .base import BasePredictor

class MyPredictor(BasePredictor):
    def __init__(self):
        super().__init__(name="my_method")
    
    def train(self, circuit, detector_samples, **kwargs):
        # 实现训练逻辑
        self.hyperedge_probs = {...}
        self.trained = True
        return {'hyperedge_probs': self.hyperedge_probs}
    
    def predict(self, circuit):
        return self.hyperedge_probs
```

### 添加新的评测器

1. 在`evaluators/`目录下创建新文件
2. 继承`BaseEvaluator`类
3. 实现`evaluate()`方法
4. 在`evaluators/__init__.py`中导入

示例：
```python
from .base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(name="my_evaluator")
    
    def evaluate(self, predicted_probs, ground_truth_probs, 
                 circuit, detector_samples, observables, **kwargs):
        # 实现评测逻辑
        return {'metric1': value1, 'metric2': value2}
```

## 数据管理

所有训练数据和结果保存在`stim_predict/data/`目录下，结构如下：

```
data/
└── <experiment_name>/
    ├── circuit.stim              # 电路文件
    ├── samples.npz               # 采样数据
    ├── metadata.json             # 元数据
    ├── predictions/              # 预测结果
    │   ├── noise_calibration.pkl
    │   ├── correlation.pkl
    │   └── rl_based.pkl
    └── evaluations/              # 评测结果
        ├── distribution_distance.json
        └── decoder_ler.json
```

## 支持的量子纠错码

- **Surface Code** (表面码): 旋转表面码，支持Z/X内存
- **Repetition Code** (重复码): 简单的重复码
- **Color Code** (颜色码): XYZ内存模式

## 注意事项

1. **计算资源**: 
   - `noise_calibration`和`correlation`方法需要计算高阶相关性，计算量较大
   - `rl_based`方法需要多轮迭代，训练时间较长
   - 建议使用多线程（通过`num_workers`参数）

2. **采样数量**:
   - 采样数量越多，相关性计算越准确，但计算时间也越长
   - 建议至少10万次采样（对于小码距）

3. **RL方法**:
   - RL方法的超参数（学习率、批次大小、训练轮数等）需要根据具体问题调整
   - 当前实现为简化版本，可以根据需要进一步优化

## 依赖库

- `stim`: 量子电路模拟
- `correlation`: 相关性计算
- `pymatching`: 解码器
- `numpy`: 数值计算
- `scipy`: 科学计算（用于距离度量）

## 常见问题

**Q: 训练时间过长怎么办？**
A: 可以减少采样次数、降低码距，或增加并行工作线程数。

**Q: 如何选择真实值（ground truth）？**
A: 默认使用DEM作为真实值。也可以使用高精度方法（如大量采样的correlation方法）作为参考。

**Q: 评测结果保存在哪里？**
A: 所有结果保存在`stim_predict/data/<experiment_name>/`目录下。

**Q: 如何比较多个方法？**
A: 使用`evaluate.py`脚本，指定多个方法名，框架会自动比较并给出排名。

