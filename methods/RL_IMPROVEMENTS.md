# RL训练代码改进说明

## 问题诊断

### 原始问题
在原始代码中，`policy_loss` 一直显示为 0.0，这是因为：

```python
baseline = np.mean(rewards)
advantages = rewards - baseline
policy_loss = -np.mean(advantages)  # 这永远等于0！
```

数学上：`np.mean(advantages) = np.mean(rewards - baseline) = np.mean(rewards) - baseline = 0`

## 改进方案（基于Sivak等人2024年论文）

### 参考论文
- **论文**: "Optimization of decoder priors for accurate quantum error correction"
- **作者**: Volodymyr Sivak, Michael Newman, and Paul Klimov (Google Quantum AI)
- **关键章节**: Appendix G - Optimization with RL

### 核心改进

#### 1. **引入Importance Ratio（重要性比率）**

论文方程(G17)定义了importance ratio:
```
χ_θθ̃(p^a) = π_θ(p^a) / π_θ̃(p^a)
```

这是PPO算法的核心，用于比较当前策略和旧策略的差异。

**实现**:
```python
# 计算当前策略和旧策略的对数概率
log_prob_new = -0.5 * sum((p - μ)² / σ²) - sum(log(σ))
log_prob_old = -0.5 * sum((p - μ_old)² / σ_old²) - sum(log(σ_old))

# Importance ratio
importance_ratio = exp(log_prob_new - log_prob_old)

# PPO clipping（论文Table I）
importance_ratio_clipped = clip(importance_ratio, 1-ε, 1+ε)
```

#### 2. **正确的Policy Loss定义**

论文方程(G23)定义了policy loss:
```
L_policy = -E[α(p) · χ_θθ̃(p)]
```

**实现**:
```python
# 标准化advantages（减少方差）
advantages_normalized = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

# Policy loss用于监控
policy_loss = -mean(advantages_normalized * importance_ratios_clipped)
```

现在这个loss **不再恒为0**，它反映了：
- advantages的大小（奖励好坏）
- importance ratio的偏离程度（策略更新幅度）

#### 3. **Baseline更新机制**

论文方程(G24)定义了baseline loss:
```
L_baseline = E[||α(p)||²]
```

**实现**:
```python
# Baseline使用指数移动平均
self.baseline = 0.9 * self.baseline + 0.1 * mean(rewards)

# Baseline loss监控（越小说明baseline越准确）
baseline_loss = mean(advantages²)
```

#### 4. **梯度计算**

论文方程(G21-G22)的梯度估计:
```
∇_θ J = E[α(p) · ∇_θ χ_θθ̃(p)]
```

对于高斯策略，均值的梯度为:
```
∇_μ log π(p|μ,σ) = (p - μ) / σ²
```

**实现**:
```python
for i in range(n_samples):
    weighted_advantage = advantages_normalized[i] * importance_ratios_clipped[i]
    grad_sample = weighted_advantage * (p[i] - μ) / σ²
    gradients += grad_sample

gradients /= n_samples
```

### 关键超参数（论文Table I）

| 超参数 | Repetition Code | Surface Code |
|--------|-----------------|--------------|
| Batch size | 70 | 70 |
| Epochs | 50 | 220 |
| Learning rate | 0.001 | 0.001 |
| Gradient clipping | 0.1 | 0.1 |
| Importance ratio clipping | 0.15 | 0.4 |
| Entropy coefficient | 0 | 0.01 |
| Initial policy std | 0.3 | 0.3 |

## 预期效果

### 训练输出变化

**修改前**:
```
Epoch 10/50: Mean Reward=3.489, Mean LER=0.000325, Policy Loss=-0.0000  ❌
Epoch 20/50: Mean Reward=3.502, Mean LER=0.000316, Policy Loss=-0.0000  ❌
```

**修改后（预期）**:
```
Epoch 10/50: Mean Reward=3.489, Mean LER=0.000325, Policy Loss=-2.3451  ✓
Epoch 20/50: Mean Reward=3.502, Mean LER=0.000316, Policy Loss=-2.4103  ✓
```

### Policy Loss的解释

- **负值较大**（如-3.0）：策略正在强化高reward的actions
- **接近0**：策略更新温和，可能接近收敛
- **正值**：可能出现问题（不应该发生）

### 收敛判断

监控以下指标：
1. **Mean Reward**: 应该逐渐增加
2. **Mean LER**: 应该逐渐减小
3. **Policy Loss**: 绝对值应该逐渐减小（表示策略稳定）
4. **Baseline Loss**: 应该逐渐减小（表示baseline准确）

## 进一步改进建议

### 1. 多Agent架构（论文核心创新）

当前实现是简化的单agent版本。论文使用多个sensor作为不同的agents：

```python
# 每个sensor是一个agent
for sensor in sensors:
    sensor_params = extract_params(sensor)
    sensor_reward = -log10(sensor_LER)
    # 计算每个agent的advantage和梯度
```

### 2. Entropy Regularization（论文方程H1）

鼓励策略探索，防止过早收敛：

```python
# 高斯策略的熵
entropy = 0.5 * sum(log(2πe * σ²))
entropy_loss = -entropy_coef * entropy
total_loss = policy_loss + value_coef * baseline_loss + entropy_loss
```

### 3. Adam优化器

论文使用Adam优化器（Table I），而不是简单的SGD：

```python
# 可以考虑使用scipy.optimize或实现简单的Adam
from scipy.optimize import minimize
```

## 测试建议

运行测试命令：
```bash
python train.py --experiment test_rl \
  --code-type surface_code \
  --distance 5 \
  --rounds 5 \
  --noise 0.001 \
  --shots 100000 \
  --methods rl_based
```

观察：
1. Policy Loss是否不再为0
2. Loss是否随训练变化
3. LER是否逐渐降低

## 参考文献

- Sivak et al. (2024). "Optimization of decoder priors for accurate quantum error correction." arXiv:2406.02700
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
- Mohamed et al. (2020). "Monte Carlo Gradient Estimation in Machine Learning." JMLR 21:1-62


