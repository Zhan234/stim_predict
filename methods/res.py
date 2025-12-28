"""基于残差进化策略的超图权重预测方法 (Residual ES)

核心思想：
1. 只微调解码器敏感边（top-K selection）
2. 使用大扰动探索（α ≈ 0.8 ~ 1.5）
3. ES优化 + rank-based reward（抗噪声）
"""

import stim
import numpy as np
from typing import Dict, Tuple, Any, List
import correlation
import os

from .base import BasePredictor
from .correlation import CorrelationPredictor

try:
    # 尝试相对导入
    from ..utils.data_manager import DataManager
except ImportError:
    # 回退到绝对导入
    import sys
    import os
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_manager import DataManager


class ResidualESPredictor(BasePredictor):
    """
    残差进化策略预测器
    
    核心优势：
    1. 边选择：只优化最敏感的top-K边
    2. 大探索：α ∈ [0.8, 1.5]，能突破LER台阶
    3. ES优化：无需梯度，rank-based reward抗噪声
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 population_size: int = 20,
                 top_k: int = 30,
                 epochs: int = 50,
                 sigma: float = 0.3,
                 learning_rate: float = 0.1,
                 use_gpu: bool = True,
                 correlation_use_numerical: bool = True,
                 correlation_num_workers: int = 16,
                 eval_frequency: int = 5,
                 num_workers: int = 8,
                 eval_subset_size: int = 10000):
        """
        初始化残差ES预测器
        
        Args:
            alpha: 调整范围控制参数（±100%即1.0，±150%即1.5）
            population_size: ES种群大小
            top_k: 选择前K个敏感边进行优化
            epochs: 训练轮数
            sigma: ES噪声标准差
            learning_rate: ES学习率
            use_gpu: 是否使用GPU（用于pymatching解码）
            correlation_use_numerical: Correlation方法是否使用数值方法
            correlation_num_workers: Correlation方法的并行线程数
            eval_frequency: 每多少轮评估一次
            num_workers: 并行工作线程数
            eval_subset_size: 评估时使用的子集大小
        """
        super().__init__(name="residual_es")
        
        # 超参数
        self.alpha = alpha
        self.population_size = population_size
        self.top_k = top_k
        self.epochs = epochs
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.correlation_use_numerical = correlation_use_numerical
        self.correlation_num_workers = correlation_num_workers
        self.eval_frequency = eval_frequency
        self.num_workers = num_workers
        self.eval_subset_size = eval_subset_size
        
        # Correlation基线
        self.correlation_predictor = None
        self.baseline_probs = None
        self.tanner_graph = None
        
        # 敏感边选择
        self.sensitive_edges = []  # top-K敏感边的索引
        self.hyperedge_list = []   # 所有超边的有序列表
        
        # ES参数
        self.theta = None  # 当前参数（只针对top-K边）
        
        # 训练历史
        self.training_history = {
            'rewards': [],
            'ler': [],
            'best_ler': []
        }
        
        # 最佳状态记录
        self._best_probs = None
        self._best_ler = None
        self._baseline_ler = None
    
    def _compute_edge_sensitivity(self, 
                                   circuit: stim.Circuit,
                                   detector_samples: np.ndarray,
                                   observables: np.ndarray,
                                   n_perturbations: int = 5) -> Dict[int, float]:
        """
        计算每个超边的敏感度（grad-free）
        
        方法：对每个边施加小扰动，测量LER变化
        
        Returns:
            {edge_idx: sensitivity_score}
        """
        print("计算超边敏感度...")
        sensitivities = {}
        
        # 使用小子集快速评估
        subset_size = min(5000, len(detector_samples))
        indices = np.random.choice(len(detector_samples), subset_size, replace=False)
        eval_samples = detector_samples[indices]
        eval_obs = observables[indices]
        
        # Baseline LER
        baseline_ler, _ = self._evaluate_ler(
            self.baseline_probs, eval_samples, eval_obs
        )
        
        for edge_idx, hyperedge in enumerate(self.hyperedge_list):
            if edge_idx % 20 == 0:
                print(f"  处理 {edge_idx}/{len(self.hyperedge_list)}...")
            
            ler_changes = []
            for _ in range(n_perturbations):
                # 扰动概率
                perturbed_probs = dict(self.baseline_probs)
                old_p = perturbed_probs[hyperedge]
                # 随机扰动 ±30%
                perturbation = np.random.uniform(-0.3, 0.3)
                new_p = np.clip(old_p * (1 + perturbation), 1e-10, 1.0)
                perturbed_probs[hyperedge] = new_p
                
                # 评估LER
                try:
                    ler, _ = self._evaluate_ler(perturbed_probs, eval_samples, eval_obs)
                    ler_change = abs(ler - baseline_ler)
                    ler_changes.append(ler_change)
                except:
                    ler_changes.append(0.0)
            
            # 敏感度 = 平均LER变化
            sensitivities[edge_idx] = np.mean(ler_changes)
        
        print(f"敏感度计算完成")
        return sensitivities
    
    def _select_sensitive_edges(self, sensitivities: Dict[int, float], top_k: int) -> List[int]:
        """选择top-K敏感边"""
        sorted_edges = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in sorted_edges[:top_k]]
        
        print(f"\n选择了top-{top_k}敏感边:")
        for i, (idx, sens) in enumerate(sorted_edges[:top_k]):
            if i < 5:  # 只打印前5个
                print(f"  #{i+1}: Edge {idx} (sensitivity={sens:.6f})")
        if top_k > 5:
            print(f"  ...")
        
        return selected
    
    def _apply_residual_actions(self, actions: np.ndarray) -> Dict[Tuple, float]:
        """
        应用残差动作到选中的边
        
        Args:
            actions: 形状 (top_k,)，范围 [-1, 1]
            
        Returns:
            更新后的超边概率字典
        """
        new_probs = dict(self.baseline_probs)
        
        for i, edge_idx in enumerate(self.sensitive_edges):
            hyperedge = self.hyperedge_list[edge_idx]
            baseline_p = self.baseline_probs[hyperedge]
            
            # P_final = P_corr * (1 + alpha * action)
            action = float(actions[i])
            new_p = baseline_p * (1.0 + self.alpha * action)
            new_p = np.clip(new_p, 1e-10, 1.0)
            new_probs[hyperedge] = float(new_p)
        
        return new_probs
    
    def _evaluate_ler(self, hyperedge_probs: Dict[Tuple, float],
                      detector_samples: np.ndarray,
                      observables: np.ndarray) -> Tuple[float, Dict]:
        """评估LER"""
        # 使用子集
        if len(detector_samples) > self.eval_subset_size:
            indices = np.random.choice(len(detector_samples), self.eval_subset_size, replace=False)
            eval_samples = detector_samples[indices]
            eval_obs = observables[indices]
        else:
            eval_samples = detector_samples
            eval_obs = observables
        
        # 构建DEM
        dem = self._build_dem_from_hyperedge_probs(hyperedge_probs)
        
        try:
            import pymatching
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predictions = matcher.decode_batch(eval_samples)
            errors = np.sum(predictions != eval_obs)
            ler = errors / len(eval_obs)
            
            metrics = {'n_errors': int(errors)}
            return ler, metrics
        except Exception as e:
            print(f"警告: LER评估失败: {e}")
            return 1.0, {}
    
    def _build_dem_from_hyperedge_probs(self, hyperedge_probs: Dict[Tuple, float]) -> stim.DetectorErrorModel:
        """从超边概率构建DEM"""
        dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in hyperedge_probs.items():
            if prob > 0:
                decompose = self.tanner_graph.stim_decompose[hyperedge]
                targets = []
                
                for line_i in range(len(decompose)):
                    h = decompose[line_i]
                    t = self.tanner_graph.hyperedge_frames
                    
                    targets += [stim.DemTarget(f"D{id_index}") for id_index in h]
                    targets += [stim.DemTarget(f"L{id_index}") for id_index in t[h]]
                    
                    if line_i != len(decompose) - 1:
                        targets.append(stim.DemTarget("^"))
                
                instruction = stim.DemInstruction("error", [prob], targets)
                dem.append(instruction)
        
        return dem
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        """使用残差ES训练"""
        observables = kwargs.get('observables', None)
        experiment_name = kwargs.get('experiment_name', None)

        print("=" * 80)
        print("步骤 1/4: 使用Correlation方法获取基线概率")
        print("=" * 80)

        # 检查是否已有correlation结果
        correlation_loaded = False
        if experiment_name:
            data_manager = DataManager()
            corr_pred_path = os.path.join(data_manager.base_dir, experiment_name, "predictions", "correlation.pkl")
            if os.path.exists(corr_pred_path):
                print(f"发现已有的correlation结果，正在加载: {corr_pred_path}")
                try:
                    corr_data = data_manager.load_prediction_results(experiment_name, "correlation")
                    self.baseline_probs = corr_data['hyperedge_probs']
                    # 从correlation数据中重建tanner_graph（如果存在的话）
                    if 'tanner_graph' in corr_data:
                        self.tanner_graph = corr_data['tanner_graph']
                        print("成功加载tanner_graph")
                        # 验证tanner_graph是否完整
                        if not hasattr(self.tanner_graph, 'hyperedge_probs') or not self.tanner_graph.hyperedge_probs:
                            print("警告: 加载的tanner_graph不完整，重新创建...")
                            self.tanner_graph = correlation.TannerGraph(circuit.detector_error_model())
                    else:
                        # 如果没有保存tanner_graph，需要重新创建
                        print("警告: correlation结果中未包含tanner_graph，正在重新创建...")
                        self.tanner_graph = correlation.TannerGraph(circuit.detector_error_model())

                    self.hyperedge_list = sorted(self.baseline_probs.keys())
                    correlation_loaded = True
                    print(f"成功加载correlation结果，共 {len(self.baseline_probs)} 个超边")
                except Exception as e:
                    print(f"加载correlation结果失败: {e}，将重新训练")
                    correlation_loaded = False
            else:
                print(f"未找到correlation结果文件: {corr_pred_path}")

        if not correlation_loaded:
            print("未找到correlation结果，开始训练correlation方法...")
            # Correlation基线
            self.correlation_predictor = CorrelationPredictor(
                use_numerical=self.correlation_use_numerical,
                num_workers=self.correlation_num_workers
            )

            corr_result = self.correlation_predictor.train(circuit, detector_samples)
            self.baseline_probs = corr_result['hyperedge_probs']
            self.tanner_graph = corr_result['tanner_graph']
            self.hyperedge_list = sorted(self.baseline_probs.keys())
        
        print(f"Correlation完成，{len(self.baseline_probs)} 个超边")
        
        # 评估baseline
        self._baseline_ler, _ = self._evaluate_ler(
            self.baseline_probs, detector_samples, observables
        )
        print(f"Baseline LER: {self._baseline_ler:.6f}")
        
        self._best_probs = dict(self.baseline_probs)
        self._best_ler = self._baseline_ler
        
        print("\n" + "=" * 80)
        print("步骤 2/4: 计算超边敏感度并选择top-K")
        print("=" * 80)
        
        # 计算敏感度
        sensitivities = self._compute_edge_sensitivity(
            circuit, detector_samples, observables, n_perturbations=5
        )
        
        # 选择敏感边
        self.sensitive_edges = self._select_sensitive_edges(sensitivities, self.top_k)
        
        print("\n" + "=" * 80)
        print("步骤 3/4: 初始化ES参数")
        print("=" * 80)
        
        # 初始化参数为0（无偏移）
        self.theta = np.zeros(len(self.sensitive_edges))
        
        print(f"ES种群大小: {self.population_size}")
        print(f"优化参数数: {len(self.sensitive_edges)}")
        print(f"调整范围α: {self.alpha} (±{self.alpha*100}%)")
        print(f"噪声σ: {self.sigma}")
        
        print("\n" + "=" * 80)
        print("步骤 4/4: ES优化")
        print("=" * 80)
        
        # ES训练循环
        for epoch in range(self.epochs):
            # 生成种群
            epsilon = np.random.randn(self.population_size, len(self.sensitive_edges))
            population = self.theta + self.sigma * epsilon
            
            # 评估种群
            rewards = []
            for i in range(self.population_size):
                actions = np.tanh(population[i])  # 限制在[-1,1]
                new_probs = self._apply_residual_actions(actions)
                
                if epoch % self.eval_frequency == 0 or i == 0:
                    ler, _ = self._evaluate_ler(new_probs, detector_samples, observables)
                else:
                    # 复用上次结果减少计算
                    ler = self.training_history['ler'][-1] if self.training_history['ler'] else self._baseline_ler
                
                # Reward = 负LER（最大化reward = 最小化LER）
                reward = -ler
                rewards.append(reward)
                
                # 更新最佳
                if ler < self._best_ler:
                    self._best_ler = ler
                    self._best_probs = new_probs
            
            rewards = np.array(rewards)
            
            # Rank-based reward（抗噪声）
            ranks = np.argsort(np.argsort(-rewards))  # 越大越好
            normalized_ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
            
            # 更新参数
            self.theta += self.learning_rate * np.dot(epsilon.T, normalized_ranks) / self.population_size
            
            # 记录
            best_reward = rewards.max()
            mean_ler = -rewards.mean()
            self.training_history['rewards'].append(float(best_reward))
            self.training_history['ler'].append(float(mean_ler))
            self.training_history['best_ler'].append(self._best_ler)
            
            # 打印进度
            improvement = (self._baseline_ler - self._best_ler) / self._baseline_ler * 100
            if epoch % 10 == 0 or self._best_ler < self.training_history['best_ler'][-2] if len(self.training_history['best_ler']) > 1 else False:
                print(f"Epoch {epoch+1}/{self.epochs}: "
                      f"Best_LER={self._best_ler:.6f} ({improvement:+.2f}%), "
                      f"Mean_LER={mean_ler:.6f}, "
                      f"Pop_std={rewards.std():.4f}")
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        print(f"Baseline LER: {self._baseline_ler:.6f}")
        print(f"最佳 LER: {self._best_ler:.6f}")
        improvement = (self._baseline_ler - self._best_ler) / self._baseline_ler * 100
        print(f"相对改进: {improvement:.2f}%")
        print(f"优化了 {len(self.sensitive_edges)}/{len(self.hyperedge_list)} 个超边")
        
        self.hyperedge_probs = self._best_probs
        self.trained = True
        
        return {
            'hyperedge_probs': self.hyperedge_probs,
            'baseline_probs': self.baseline_probs,
            'baseline_ler': self._baseline_ler,
            'final_mean_ler': self._best_ler,
            'final_mean_reward': self.training_history['rewards'][-1],
            'improvement_percent': improvement,
            'training_history': self.training_history,
            'tanner_graph': self.tanner_graph,
            'sensitive_edges': self.sensitive_edges
        }
    
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        return self.hyperedge_probs
    
    def get_detector_error_model(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        return self._build_dem_from_hyperedge_probs(self.hyperedge_probs)


# 保持向后兼容的别名
ResidualRLPredictor = ResidualESPredictor
