"""基于相关性分析的超图权重预测方法"""

import stim
import numpy as np
from typing import Dict, Tuple, Any
import correlation

from .base import BasePredictor


class CorrelationPredictor(BasePredictor):
    """
    相关性分析预测器
    
    直接使用correlation库从DEM分析得到理想的相关性
    参考: repetition_code.py 和 surface_code.py
    """
    
    def __init__(self, use_numerical: bool = True, num_workers: int = 16):
        super().__init__(name="correlation")
        self.use_numerical = use_numerical
        self.num_workers = num_workers
        self.tanner_graph = None
    
    def train(self, circuit: stim.Circuit, detector_samples: np.ndarray, **kwargs) -> Dict:
        # 调试：检查detector_samples中是否有错误
        print(f"Detector samples shape: {detector_samples.shape}")
        print(f"Detector samples dtype: {detector_samples.dtype}")
        print(f"Detector samples range: [{detector_samples.min()}, {detector_samples.max()}]")
        print(f"Detector samples non-zero count: {np.count_nonzero(detector_samples)}")
        print(f"Detector samples error rate: {np.count_nonzero(detector_samples) / detector_samples.size:.2e}")

        # 获取DEM
        decompose = kwargs.get('decompose_errors', True)
        approximate = kwargs.get('approximate_disjoint_errors', True)
        dem = circuit.detector_error_model(decompose_errors=decompose, approximate_disjoint_errors=approximate)
        # 默认的 fallback 概率：不要回退到 ground-truth DEM 的概率，使用一个很小的默认值
        default_fallback_prob = float(kwargs.get('default_fallback_prob', 1e-6))
        
        if self.use_numerical:
            # 数值方法：使用高阶相关性分析
            self.tanner_graph = correlation.TannerGraph(dem)
            
            # 计算高阶相关性
            result = correlation.cal_high_order_correlations(
                detector_samples, 
                self.tanner_graph.hyperedges, 
                num_workers=self.num_workers
            )
            
            # 收集超边概率
            hyperedge_probs = {}
            for hyperedge, prob_dem in self.tanner_graph.hyperedge_probs.items():
                prob_corr = result.get(hyperedge)
                # 使用相关性计算的概率；如果无效则使用小的默认概率（不要泄露 DEM 的真实概率）
                if prob_corr is not None and prob_corr > 0:
                    hyperedge_probs[hyperedge] = prob_corr
                else:
                    hyperedge_probs[hyperedge] = default_fallback_prob
        
        else:
            # 解析方法：使用二阶相关性（仅适用于简单码如重复码）
            result = correlation.cal_2nd_order_correlations(detector_samples)
            bdy, edges = result.data
            # 规范 bdy/edges：如果是数组，取均值作为代表性的标量值；如果是 dict，则保留原样用于按超边索引查找
            def _to_scalar_or_dict(x):
                if x is None:
                    return None
                if isinstance(x, dict):
                    return x
                try:
                    arr = np.asarray(x)
                except Exception:
                    return x
                if arr.size == 1:
                    return float(arr.item())
                # 多元素数组 -> 使用均值作为回退的标量估计
                return float(np.mean(arr))

            bdy_val = _to_scalar_or_dict(bdy)
            edges_val = _to_scalar_or_dict(edges)
            # 不再使用 ground-truth DEM 给出的理想相关性作为直接回退值
            bdy_ideal, edges_ideal = correlation.correlation_from_detector_error_model(dem)
            self.tanner_graph = correlation.TannerGraph(dem)
            hyperedge_probs = {}
            
            # 使用计算得到的相关性来调整DEM中的概率
            for hyperedge, prob_dem in self.tanner_graph.hyperedge_probs.items():
                # 根据超边的阶数选择使用哪个相关性
                hyperedge_order = len(hyperedge)
                
                if hyperedge_order == 1:
                    if bdy_val is not None and isinstance(bdy_val, (float, int)) and bdy_val > 0:
                        hyperedge_probs[hyperedge] = bdy_val
                    else:
                        hyperedge_probs[hyperedge] = default_fallback_prob
                
                elif hyperedge_order == 2:
                    if edges is not None:
                        # 如果edges是字典，尝试查找对应的边
                        if isinstance(edges_val, dict) and hyperedge in edges_val:
                            prob_corr = edges_val[hyperedge]
                            if prob_corr is not None and prob_corr > 0:
                                hyperedge_probs[hyperedge] = prob_corr
                            else:
                                hyperedge_probs[hyperedge] = default_fallback_prob
                        # 如果edges是单个标量值（适用于重复码的均匀情况）
                        elif isinstance(edges_val, (float, int)) and edges_val > 0:
                            hyperedge_probs[hyperedge] = edges_val
                        else:
                            hyperedge_probs[hyperedge] = default_fallback_prob
                    else:
                        hyperedge_probs[hyperedge] = default_fallback_prob
                
                else:
                    hyperedge_probs[hyperedge] = default_fallback_prob
            
            # 保存相关性信息用于调试
            self._correlation_info = {
                'bdy_calculated': bdy,
                'edges_calculated': edges,
                'bdy_used': bdy_val,
                'edges_used': edges_val,
                'bdy_ideal': bdy_ideal,
                'edges_ideal': edges_ideal
            }
        
        self.hyperedge_probs = hyperedge_probs
        self.trained = True

        # 添加调试信息
        print(f"Correlation predictor training completed:")
        print(f"  - Use numerical: {self.use_numerical}")
        print(f"  - Number of hyperedges: {len(hyperedge_probs)}")
        print(f"  - Probability range: [{min(hyperedge_probs.values()):.2e}, {max(hyperedge_probs.values()):.2e}]")
        print(f"  - Non-zero probabilities: {sum(1 for p in hyperedge_probs.values() if p > 0)}")

        return {
            'hyperedge_probs': hyperedge_probs,
            'tanner_graph': self.tanner_graph,
            'correlation_info': self._correlation_info if hasattr(self, '_correlation_info') else None
        }
    
    def predict(self, circuit: stim.Circuit) -> Dict[Tuple, float]:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        return self.hyperedge_probs
    
    def get_detector_error_model(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        if not self.trained:
            raise RuntimeError("预测器尚未训练")
        
        # 构建新的DEM
        new_dem = stim.DetectorErrorModel()
        
        for hyperedge, prob in self.hyperedge_probs.items():
            if prob > 0:
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

