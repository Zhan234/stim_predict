"""基于Actor-Critic算法的超图权重预测方法"""

import stim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, List, Optional
import correlation
from copy import deepcopy

from .base import BasePredictor


class ActorNetwork(nn.Module):
    """Actor网络：输出增量（Δlog p）的均值和标准差，仅作用于Top-K超边。"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 输出增量均值（用tanh限制范围，较小步长防止破坏性更新）
        mean = torch.tanh(self.mean(x)) * 0.05  # [-0.05, 0.05]
        log_std = torch.clamp(self.log_std(x), -5, -2)  # 更小的方差，稳定探索
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic网络：仅作为基线估计器，输出即时价值（不建模长期回报）。"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class ACPredictor(BasePredictor):
    """
    基于Actor-Critic算法的预测器
    
    使用Actor-Critic强化学习算法优化decoder priors
    - Actor: 学习策略，输出超边概率的分布
    - Critic: 学习价值函数V(s)，用于计算优势函数
    - 使用TD误差作为优势估计
    
    核心思想:
    1. Actor网络输出超边概率的**增量**（Δlog(p)），而非绝对值
    2. Critic网络评估当前状态的价值
    3. 使用TD误差更新Critic，使用优势函数更新Actor
    4. Reward定义为相对于初始DEM的改进度（更稳定）
    5. Action是增量：s_{t+1} = s_t + α * action（而非s_{t+1} = action）
    """
    
    def __init__(self,
                 learning_rate_actor: float = 1e-4,
                 learning_rate_critic: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100,
                 hidden_dim: int = 128,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 use_gpu: bool = True,
                 eval_frequency: int = 10,
                 action_scale: float = 0.01,
                 top_k: int = 16,
                 reward_ema_beta: float = 0.9,
                 accept_tolerance: float = 0.02):
        """
        初始化Actor-Critic预测器
        
        Args:
            learning_rate_actor: Actor网络学习率
            learning_rate_critic: Critic网络学习率（通常比Actor大）
            batch_size: 批次大小
            epochs: 训练轮数
            hidden_dim: 隐藏层维度
            gamma: 折扣因子（用于TD误差计算）
            entropy_coef: 熵损失系数（鼓励探索）
            max_grad_norm: 梯度裁剪的最大范数
            use_gpu: 是否使用GPU
            eval_frequency: 每多少步真正评估一次LER（其余时间使用缓存）
            action_scale: 动作增量的缩放因子（控制更新步长）
            accept_tolerance: LER容忍度，超过则回滚
        """
        super().__init__(name="actor_critic")
        
        # 超参数
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.eval_frequency = eval_frequency
        self.action_scale = action_scale
        self.top_k = top_k
        self.reward_ema_beta = reward_ema_beta
        self.accept_tolerance = accept_tolerance
        
        # 设备设置
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        if self.use_gpu:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU")
        
        # Actor和Critic网络（延迟初始化，等待知道维度）
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        
        # Tanner图
        self.tanner_graph = None
        self.top_k_indices: List[int] = []
        
        # 状态表示（当前超边概率的对数）
        self.state = None

        # 奖励平滑
        self._ema_ler = None
        
        # 缓存机制
        self._cached_matcher = None
        self._cached_dem_key = None
        self._last_ler = None
        self._ler_cache = {}  # 策略参数 -> LER的缓存
        
        # 基准LER（用于计算相对改进）
        self._baseline_ler = None
        self._initial_dem = None
        self._best_state = None
        self._best_ler = None
        
        # 训练历史
        self.training_history = {
            'rewards': [],
            'ler': [],
            'actor_loss': [],
            'critic_loss': [],
            'td_error': []
        }
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        """
        使用Actor-Critic训练预测器
        
        Args:
            circuit: Stim电路对象
            detector_samples: 探测器采样数据
            **kwargs: 其他参数，可包含:
                - observables: 观测量数据
                - decoder_type: 解码器类型，默认'pymatching'
                
        Returns:
            训练结果字典
        """
        observables = kwargs.get('observables', None)
        decoder_type = kwargs.get('decoder_type', 'pymatching')
        
        # 获取原始DEM和Tanner图
        dem_origin = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True
        )
        self.tanner_graph = correlation.TannerGraph(dem_origin)
        self._initial_dem = dem_origin
        
        # 计算baseline LER（原始DEM的性能）
        print("计算baseline LER...")
        self._baseline_ler = self._evaluate_ler_with_cache(
            dem_origin,
            detector_samples,
            observables,
            decoder_type
        )
        print(f"Baseline LER: {self._baseline_ler:.6f}")
        
        # 初始化网络
        n_params = len(self.tanner_graph.hyperedge_probs)
        # 选择Top-K关键超边（概率最大的K个）
        sorted_edges = sorted(self.tanner_graph.hyperedge_probs.items(), key=lambda x: x[1], reverse=True)
        k = min(self.top_k, n_params)
        self.top_k_indices = [list(self.tanner_graph.hyperedge_probs.keys()).index(e[0]) for e in sorted_edges[:k]]

        state_dim = k              # 仅感知Top-K超边的状态
        action_dim = k             # 仅对Top-K超边输出增量
        
        self.actor = ActorNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, self.hidden_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic
        )
        
        # 初始化状态：使用平坦的小概率，而不是DEM概率，避免先验过强
        init_probs = np.ones(n_params) * 1e-3
        init_probs = np.clip(init_probs, 1e-7, 0.49)
        self.state = torch.tensor(
            np.log(init_probs),
            dtype=torch.float32,
            device=self.device
        )

        # 最优状态初始化为baseline对应的状态
        self._best_state = self.state.clone().detach()
        self._best_ler = self._baseline_ler
        
        print(f"开始Actor-Critic训练，共 {self.epochs} 轮，批次大小 {self.batch_size}")
        print(f"Actor学习率: {self.learning_rate_actor}, Critic学习率: {self.learning_rate_critic}")
        print(f"动作缩放: {self.action_scale}")
        
        # 初始化上次LER（使用baseline）
        self._last_ler = self._baseline_ler
        self._ema_ler = self._baseline_ler
        
        # Actor-Critic训练循环
        for epoch in range(self.epochs):
            # 收集经验
            states, actions, rewards, log_probs_list = [], [], [], []
            lers = []
            candidates_batch = []
            
            for step_i in range(self.batch_size):
                # 1. 仅取Top-K的状态输入Actor
                top_k_state = self.state[self.top_k_indices].unsqueeze(0)
                with torch.no_grad():
                    mean, log_std = self.actor(top_k_state)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action_delta = dist.sample()  # 仅作用于Top-K
                    action_delta = torch.clamp(action_delta, -1.0, 1.0)
                    log_prob = dist.log_prob(action_delta).sum()
                
                # 2. 应用增量到Top-K，其余不变
                prev_state = self.state.clone().detach()
                new_log_probs = self.state.clone()
                new_log_probs[self.top_k_indices] = new_log_probs[self.top_k_indices] + self.action_scale * action_delta.squeeze(0)
                new_log_probs = torch.clamp(new_log_probs, np.log(1e-7), np.log(0.49))
                new_probs = torch.exp(new_log_probs)
                
                hyperedge_probs_candidate = {}
                new_probs_np = new_probs.cpu().numpy()
                for j, hyperedge in enumerate(self.tanner_graph.hyperedge_probs.keys()):
                    hyperedge_probs_candidate[hyperedge] = float(new_probs_np[j])
                
                candidates_batch.append({
                    'hyperedge_probs': hyperedge_probs_candidate,
                    'state_topk': top_k_state.cpu().numpy().squeeze(0),
                    'action_topk': action_delta.squeeze(0).cpu().numpy(),
                    'log_prob': log_prob,
                    'next_state': new_log_probs.detach()
                })
                
                # 更新全局状态（作为下一步的起点）
                self.state = new_log_probs.detach()
            
            # 3. 评估LER（真实评估，但后续做平滑奖励）
            batch_lers = self._evaluate_ler_batch(
                [c['hyperedge_probs'] for c in candidates_batch],
                detector_samples,
                observables,
                decoder_type
            )
            
            # 4. 奖励：对LER做指数滑动，稳定梯度；reward = -log10(EMA_LER)
            for i, candidate in enumerate(candidates_batch):
                ler = batch_lers[i]
                # 记录最优checkpoint（不回滚当前state）
                if ler < self._best_ler:
                    self._best_ler = ler
                    self._best_state = candidate['next_state'].clone().detach()

                # 奖励：相对baseline的改进度
                self._ema_ler = self.reward_ema_beta * self._ema_ler + (1 - self.reward_ema_beta) * ler
                safe_ler = max(self._ema_ler, 1e-10)
                reward = np.log10(self._baseline_ler / safe_ler)
                
                states.append(candidate['state_topk'])
                actions.append(candidate['action_topk'])
                rewards.append(reward)
                log_probs_list.append(candidate['log_prob'])
                lers.append(ler)
            
            # 张量化（仅Top-K维度）
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            log_probs_tensor = torch.stack(log_probs_list)
            
            # 5. Critic作为基线：value ≈ reward，单步回报
            values = self.critic(states_tensor)
            critic_loss = F.mse_loss(values, rewards_tensor)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # 6. Actor：优势 = reward - value
            with torch.no_grad():
                advantages = rewards_tensor - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            actor_loss_total = 0
            entropy_total = 0
            for i in range(self.batch_size):
                mean, log_std = self.actor(states_tensor[i:i+1])
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action_tensor = torch.tensor(
                    actions[i],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)
                log_prob = dist.log_prob(action_tensor).sum()
                entropy = dist.entropy().sum()
                actor_loss = -(log_prob * advantages[i].detach())
                actor_loss_total += actor_loss
                entropy_total += entropy
            actor_loss_total = actor_loss_total / self.batch_size
            entropy_total = entropy_total / self.batch_size
            total_actor_loss = actor_loss_total - self.entropy_coef * entropy_total
            
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 记录训练历史
            mean_reward = np.mean(rewards)
            mean_ler = np.mean(lers)
            mean_td_error = advantages.abs().mean().item()
            
            self.training_history['rewards'].append(mean_reward)
            self.training_history['ler'].append(mean_ler)
            self.training_history['actor_loss'].append(total_actor_loss.item())
            self.training_history['critic_loss'].append(critic_loss.item())
            self.training_history['td_error'].append(mean_td_error)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                improvement_pct = (self._baseline_ler - mean_ler) / self._baseline_ler * 100
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Reward={mean_reward:.3f}, "
                      f"LER={mean_ler:.6f} (改进{improvement_pct:+.1f}%), "
                      f"Actor Loss={total_actor_loss.item():.4f}, "
                      f"Critic Loss={critic_loss.item():.4f}, "
                      f"TD Error={mean_td_error:.4f}")
        
        # 使用最终状态作为预测
        # 回滚到最优checkpoint，防止末期崩盘（仅在训练结束时应用）
        if self._best_state is not None:
            self.state = self._best_state.clone().detach()
        final_probs = torch.exp(self.state).clamp(1e-7, 0.49).cpu().numpy()
        
        final_hyperedge_probs = {}
        for i, hyperedge in enumerate(self.tanner_graph.hyperedge_probs.keys()):
            final_hyperedge_probs[hyperedge] = float(final_probs[i])
        
        self.hyperedge_probs = final_hyperedge_probs
        self.trained = True
        
        return {
            'hyperedge_probs': final_hyperedge_probs,
            'training_history': self.training_history,
            'final_mean_reward': self.training_history['rewards'][-1],
            'final_mean_ler': self.training_history['ler'][-1]
        }
    
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        """
        预测超边概率
        
        Args:
            circuit: Stim电路对象
            
        Returns:
            超边到概率的映射字典
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self.hyperedge_probs
    
    def _build_dem_from_hyperedge_probs(self, hyperedge_probs: Dict[Tuple, float]) -> stim.DetectorErrorModel:
        """
        从超边概率构建DEM
        
        Args:
            hyperedge_probs: 超边概率字典
            
        Returns:
            探测器错误模型
        """
        new_dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            # 限制概率范围：不能太小（避免pymatching权重溢出）也不能太大
            # 必须小于0.5，否则pymatching会报错
            prob = float(np.clip(prob, 1e-7, 0.49))
            if prob > 1e-7:
                decompose = self.tanner_graph.stim_decompose[hyperedge]
                targets = []
                
                for line_i in range(len(decompose)):
                    h = decompose[line_i]
                    t = self.tanner_graph.hyperedge_frames
                    
                    # 添加探测器目标
                    targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                    # 添加逻辑观测量目标
                    targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                    
                    # 添加分隔符
                    if line_i != len(decompose) - 1:
                        targets.append(stim.DemTarget("^"))
                
                instruction = stim.DemInstruction("error", [prob], targets)
                new_dem.append(instruction)
        
        return new_dem
    
    def _evaluate_ler_batch(self,
                           hyperedge_probs_list: List[Dict[Tuple, float]],
                           detector_samples: np.ndarray,
                           observables: Optional[np.ndarray],
                           decoder_type: str = 'pymatching') -> List[float]:
        """
        批量评估多个候选的LER（更高效）
        
        Args:
            hyperedge_probs_list: 多个候选的超边概率字典列表
            detector_samples: 探测器采样数据
            observables: 观测量数据
            decoder_type: 解码器类型
            
        Returns:
            每个候选的逻辑错误率列表
        """
        lers = []
        for hyperedge_probs in hyperedge_probs_list:
            dem = self._build_dem_from_hyperedge_probs(hyperedge_probs)
            ler = self._evaluate_ler_with_cache(
                dem,
                detector_samples,
                observables,
                decoder_type
            )
            lers.append(ler)
        return lers
    
    def _evaluate_ler_with_cache(self,
                                 dem: stim.DetectorErrorModel,
                                 detector_samples: np.ndarray,
                                 observables: Optional[np.ndarray],
                                 decoder_type: str = 'pymatching') -> float:
        """
        评估给定DEM的逻辑错误率（带缓存优化）
        
        Args:
            dem: 探测器错误模型
            detector_samples: 探测器采样数据
            observables: 观测量数据
            decoder_type: 解码器类型
            
        Returns:
            逻辑错误率
        """
        if decoder_type != 'pymatching':
            raise NotImplementedError(f"不支持的解码器类型: {decoder_type}")
        
        try:
            import pymatching
        except ImportError:
            raise ImportError("需要安装pymatching库: pip install pymatching")
        
        # 生成DEM的key用于缓存（简化版：使用str表示）
        dem_key = str(dem)[:1000]  # 截断以避免太长
        
        # 检查是否可以复用缓存的matcher
        if self._cached_matcher is None or self._cached_dem_key != dem_key:
            # 构建新的matcher
            self._cached_matcher = pymatching.Matching.from_detector_error_model(dem)
            self._cached_dem_key = dem_key
        
        # 使用缓存的matcher解码（如果DEM结构相同，这会快很多）
        matcher = self._cached_matcher
        
        # 解码
        predictions = matcher.decode_batch(detector_samples)
        
        # 计算逻辑错误率
        if observables is not None:
            errors = np.any(predictions != observables, axis=1)
            ler = np.mean(errors)
        else:
            # 如果没有观测量，假设全0为正确结果
            errors = np.any(predictions != 0, axis=1)
            ler = np.mean(errors)
        
        return float(ler)
    
    def _evaluate_ler(self,
                     dem: stim.DetectorErrorModel,
                     detector_samples: np.ndarray,
                     observables: Optional[np.ndarray],
                     decoder_type: str = 'pymatching') -> float:
        """
        评估给定DEM的逻辑错误率（向后兼容的包装）
        
        Args:
            dem: 探测器错误模型
            detector_samples: 探测器采样数据
            observables: 观测量数据
            decoder_type: 解码器类型
            
        Returns:
            逻辑错误率
        """
        return self._evaluate_ler_with_cache(dem, detector_samples, observables, decoder_type)
    
    def get_detector_error_model(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        """
        获取基于预测概率的探测器错误模型
        
        Args:
            circuit: Stim电路对象
            
        Returns:
            更新后的探测器错误模型
        """
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self._build_dem_from_hyperedge_probs(self.hyperedge_probs)

