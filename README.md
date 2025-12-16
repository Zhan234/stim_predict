# 超图权重预测评测框架

用于评测量子纠错中不同超图权重预测方法性能的框架。

## 功能概述

本框架实现了三种超图权重预测方法，并提供了两类评测指标：

### 预测方法

1. **correlation**: 基于相关性分析的方法，支持数值方法（高阶相关性）和解析方法（二阶相关性，适用于重复码等简单拓扑）
2. **rl_based**: 基于强化学习（PPO）的方法，通过优化decoder priors来最小化逻辑错误率
3. **grpo**: 基于GRPO（Group Relative Policy Optimization）的方法，去除critic模型，使用组内相对奖励作为优势估计，更高效且内存友好
4. **actor_critic (ac)**: 基于Actor-Critic算法的方法，使用独立的Actor和Critic网络，通过TD误差学习价值函数并优化策略，更稳定的梯度估计

### 评测指标

1. **distribution_distance**: 预测概率分布与真实概率分布之间的距离（MAE, RMSE, KL散度等）
2. **decoder_ler**: 不同解码器（PyMatching, PyMatching with correlations等）下的逻辑错误率

## 目录结构

```
stim_predict/
├── README.md                    # 本文档
├── config.json                  # 配置文件示例/默认配置
├── methods/                     # 预测方法实现
│   ├── base.py                  # 基类定义
│   ├── correlation.py           # 相关性方法
│   ├── rl_based.py              # 强化学习方法（PPO）
│   ├── grpo.py                  # GRPO方法
│   └── ac.py                    # Actor-Critic方法
├── circuits/                    # 电路生成
│   └── circuit_factory.py       # 电路工厂（支持surface code, repetition code等）
├── evaluators/                  # 评测器
│   ├── base.py                  # 基类定义
│   ├── distribution_distance.py # 概率分布距离评测
│   └── decoder_ler.py           # 解码器LER评测
├── utils/                       # 工具函数
│   └── data_manager.py          # 数据管理（保存/加载训练数据和结果）
├── data/                        # 数据存储目录
├── train.py                     # 训练脚本（支持配置文件和命令行参数）
├── evaluate.py                 # 评测脚本
├── visualize.py                 # 结果可视化工具
├── example.py                   # 使用示例
└── requirements.txt             # 依赖列表
```

## 快速开始

### 1. 简单示例

```python
from circuits import CircuitFactory
from methods import CorrelationPredictor
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
predictor = CorrelationPredictor(use_numerical=True, num_workers=4)
result = predictor.train(circuit, detector_samples)

# 评测
evaluator = DistributionDistanceEvaluator()
# ... (详见 example.py)
```

### 2. 使用命令行工具

#### 训练

训练可以通过两种方式配置参数：**配置文件**或**命令行参数**。命令行参数的优先级高于配置文件。

**方式1：使用配置文件（推荐）**

创建或编辑 `config.json`：

```json
{
  "experiment_name": "my_exp",
  "circuit": { "code_type": "surface_code", "distance": 5, "rounds": 5, "noise_level": 0.001 },
  "sampling": { "n_shots": 100000 },
  "training": {
    "num_workers": 8,
    "methods": ["correlation", "rl_based", "grpo", "actor_critic"],
    "correlation": {
      "use_numerical": true,
      "num_workers": 16
    },
    "rl_based": {
      "learning_rate": 0.001,
      "epochs": 50,
      "batch_size": 32,
      "clip_ratio": 0.2,
      "entropy_coef": 0.01,
      "value_coef": 0.5,
      "max_grad_norm": 0.5,
      "use_gpu": true
    },
    "grpo": {
      "learning_rate": 1e-6,
      "epochs": 50,
      "group_size": 64,
      "clip_ratio": 0.2,
      "kl_coef": 0.04,
      "max_grad_norm": 0.5,
      "supervision_mode": "outcome",
      "use_gpu": true
    },
    "ac": {
      "learning_rate_actor": 1e-4,
      "learning_rate_critic": 1e-3,
      "epochs": 100,
      "batch_size": 32,
      "hidden_dim": 128,
      "gamma": 0.99,
      "entropy_coef": 0.01,
      "max_grad_norm": 0.5,
      "use_gpu": true,
      "eval_frequency": 10
    }
  },
  "evaluation": {
    "methods": ["correlation", "rl_based", "grpo", "actor_critic"],
    "evaluators": ["distribution_distance", "decoder_ler"],
    "ground_truth": "dem",
    "decoders": ["pymatching", "pymatching_corr"],
    "test_shots": 100000
  }
}
```

然后运行：

```bash
cd stim_predict
python train.py --config config.json
```

**方式2：使用命令行参数**

```bash
cd stim_predict
python train.py --experiment my_exp --code-type surface_code --distance 5 --rounds 5 --noise 0.001 --shots 100000 --methods correlation rl_based grpo actor_critic
```

**方式3：混合使用（命令行参数覆盖配置文件）**

```bash
python train.py --config config.json --experiment my_custom_exp --distance 7
```

参数说明：
- `--config`: 配置文件路径（JSON格式，可选）
- `--experiment`: 实验名称（必需，或从配置文件读取）
- `--code-type`: 编码类型（surface_code, repetition_code, color_code）
- `--distance`: 码距
- `--rounds`: 测量轮数
- `--noise`: 噪声水平
- `--shots`: 采样次数
- `--methods`: 要训练的方法列表
- `--workers`: 并行工作线程数
- `--rl-epochs`: RL方法的训练轮数（仅用于rl_based方法）
- `--rl-batch-size`: RL方法的批次大小（仅用于rl_based方法）
- `--grpo-epochs`: GRPO方法的训练轮数（仅用于grpo方法）
- `--grpo-group-size`: GRPO方法的组大小（仅用于grpo方法）
- `--ac-epochs`: Actor-Critic方法的训练轮数（仅用于ac方法）
- `--ac-batch-size`: Actor-Critic方法的批次大小（仅用于ac方法）
- 更多超参（如Correlation的use_numerical/num_workers，RL/GRPO/AC的learning_rate、clip_ratio、kl_coef、max_grad_norm、use_gpu、supervision_mode、gamma、entropy_coef、hidden_dim等）可在 `config.json` 的 `training` 下各自子块配置

**参数优先级**：命令行参数 > 配置文件参数 > 默认值

#### 评测

```bash
# 使用配置文件
python evaluate.py --config config.json

# 覆盖部分参数
python evaluate.py --config config.json --methods correlation grpo --test-shots 200000
```

参数说明：
- `--experiment`: 实验名称（必需）
- `--methods`: 要评测的方法列表
- `--evaluators`: 要使用的评测器（distribution_distance, decoder_ler）
- `--ground-truth`: 真实值来源（默认为'dem'）
- `--decoders`: 要测试的解码器列表（用于decoder_ler评测器）
- `--test-shots`: 测试集采样次数（默认100000，独立于训练数据）
- 也可在 `config.json` 的 `evaluation` 部分配置以上评测参数

#### 可视化结果

```bash
python visualize.py --experiment my_exp           # 查看实验结果
python visualize.py --list                        # 列出所有实验
```



## 使用流程

### 训练阶段

1. 创建量子纠错码电路（使用`CircuitFactory`）
2. 采样训练数据（探测器数据）
3. 训练各个预测方法
4. 保存训练数据、ground truth DEM和预测结果（使用`DataManager`）

### 评测阶段

1. 加载实验配置和预测结果
2. **重新采样独立的测试数据**（确保泛化性）
3. 加载保存的ground truth DEM作为真实值
4. 运行评测器比较预测结果与真实值
5. 保存并分析评测结果

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
    ├── ground_truth.dem          # Ground truth DEM（用于评测的真实值）
    ├── samples.npz               # 训练数据采样
    ├── metadata.json             # 元数据
    ├── predictions/              # 预测结果
    │   ├── correlation.pkl
    │   ├── rl_based.pkl
    │   └── grpo.pkl
    └── evaluations/              # 评测结果
        ├── distribution_distance.json
        └── decoder_ler.json
```

**注意**：评测时会重新从ground truth电路采样独立的测试数据，确保评测结果的泛化性。

## 支持的量子纠错码

- **Surface Code** (表面码): 旋转表面码，支持Z/X内存
- **Repetition Code** (重复码): 简单的重复码
- **Color Code** (颜色码): XYZ内存模式

## 配置文件使用

框架支持通过JSON配置文件来管理训练参数，便于实验管理和复现。

### 配置文件格式

参考 `config.json`，配置文件包含以下主要部分：

- `experiment_name`: 实验名称
- `circuit`: 电路参数（code_type, distance, rounds, noise_level）
- `sampling`: 采样参数（n_shots）
- `training`: 训练参数（num_workers, methods, 各方法的超参数）
- `methods`: 各方法的详细配置（可选）

### 使用配置文件

```bash
# 使用配置文件
python train.py --config config.json

# 配置文件 + 命令行参数覆盖
python train.py --config config.json --experiment custom_name --distance 7
```

### 参数优先级

1. **命令行参数**（最高优先级）
2. **配置文件参数**
3. **默认值**（最低优先级）

### 配置文件示例

```json
{
  "experiment_name": "my_exp",
  "circuit": {
    "code_type": "surface_code",
    "distance": 5,
    "rounds": 5,
    "noise_level": 0.001
  },
  "sampling": {
    "n_shots": 100000
  },
  "training": {
    "num_workers": 8,
    "methods": ["correlation", "rl_based", "grpo"],
    "rl_based": {
      "epochs": 50,
      "batch_size": 32
    },
    "grpo": {
      "epochs": 100,
      "group_size": 64
    },
    "ac": {
      "epochs": 100,
      "batch_size": 32,
      "learning_rate_actor": 1e-4,
      "learning_rate_critic": 1e-3,
      "eval_frequency": 10
    }
  }
}
```

## 注意事项

1. **计算资源**: 
   - `correlation`方法需要计算高阶相关性，计算量较大
   - `rl_based`和`grpo`方法需要多轮迭代，训练时间较长
   - 建议使用多线程（通过`num_workers`参数）
   - `grpo`方法支持GPU加速，建议使用GPU训练

2. **采样数量**:
   - 采样数量越多，相关性计算越准确，但计算时间也越长
   - 建议至少10万次采样（对于小码距）

3. **RL方法**:
   - RL方法的超参数（学习率、批次大小、训练轮数等）需要根据具体问题调整
   - `grpo`方法相比`rl_based`方法内存效率更高，推荐使用
   - `actor_critic`方法使用独立的Actor和Critic网络，提供更稳定的梯度估计，适合需要精确价值函数的场景
   - AC方法使用了性能优化：通过`eval_frequency`参数控制LER评估频率（默认每10轮评估一次），大幅提升训练速度
   - 当前实现为简化版本，可以根据需要进一步优化

4. **配置文件**:
   - 使用配置文件可以方便地管理多个实验的参数
   - 命令行参数可以覆盖配置文件中的对应参数
   - 建议为每个实验创建独立的配置文件

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

