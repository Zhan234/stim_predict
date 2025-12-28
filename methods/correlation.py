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

            print("相关性分析结果:")
            print(f"边界相关性形状: {np.array(bdy).shape}")
            print(f"边相关性形状: {np.array(edges).shape}")

            # 不再使用 ground-truth DEM 给出的理想相关性作为直接回退值
            bdy_ideal, edges_ideal = correlation.correlation_from_detector_error_model(dem)
            self.tanner_graph = correlation.TannerGraph(dem)
            hyperedge_probs = {}

            # 使用计算得到的相关性来调整DEM中的概率
            for hyperedge, prob_dem in self.tanner_graph.hyperedge_probs.items():
                hyperedge_order = len(hyperedge)

                if hyperedge_order == 1:
                    # 单检测器错误：使用边界相关性
                    detector_idx = list(hyperedge)[0]
                    if hasattr(bdy, '__len__') and len(bdy) > detector_idx:
                        prob = bdy[detector_idx]
                    else:
                        prob = default_fallback_prob

                elif hyperedge_order == 2:
                    # 双检测器错误：使用边相关性
                    detector_indices = list(hyperedge)
                    if len(detector_indices) == 2:
                        i, j = detector_indices
                        if hasattr(edges, 'shape') and edges.shape[0] > i and edges.shape[1] > j:
                            prob = edges[i, j]
                        else:
                            prob = default_fallback_prob
                    else:
                        prob = default_fallback_prob

                else:
                    # 高阶错误：使用小的默认概率
                    prob = default_fallback_prob

                # 确保概率在合理范围内
                prob = np.clip(prob, 1e-10, 1.0)
                hyperedge_probs[hyperedge] = prob
            
            # 保存相关性信息用于调试
            self._correlation_info = {
                'bdy_calculated': bdy,
                'edges_calculated': edges,
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

