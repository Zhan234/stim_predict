"""概率分布距离评测器"""

import numpy as np
from typing import Dict, Any, Tuple
import stim
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from .base import BaseEvaluator


class DistributionDistanceEvaluator(BaseEvaluator):
    """
    概率分布距离评测器
    
    计算预测的超边概率分布与真实概率分布之间的各种距离度量
    """
    
    def __init__(self):
        """初始化分布距离评测器"""
        super().__init__(name="distribution_distance")
    
    def evaluate(self,
                predicted_probs: Dict,
                ground_truth_probs: Dict,
                circuit: stim.Circuit = None,
                detector_samples: np.ndarray = None,
                observables: np.ndarray = None,
                **kwargs) -> Dict[str, Any]:
        """
        评测预测概率分布与真实概率分布的距离
        
        Args:
            predicted_probs: 预测的超边概率字典
            ground_truth_probs: 真实的超边概率字典
            circuit: Stim电路对象（可选）
            detector_samples: 探测器采样数据（可选）
            observables: 观测量真值（可选）
            **kwargs: 其他参数
            
        Returns:
            包含各种距离度量的字典
        """
        # 对齐两个字典的键
        common_keys = set(predicted_probs.keys()) & set(ground_truth_probs.keys())
        
        if len(common_keys) == 0:
            return {
                'error': '预测结果与真实值没有共同的超边',
                'n_predicted': len(predicted_probs),
                'n_ground_truth': len(ground_truth_probs)
            }
        
        # 提取对应的概率值
        pred_vals = np.array([predicted_probs[k] for k in common_keys])
        true_vals = np.array([ground_truth_probs[k] for k in common_keys])
        
        # 计算各种距离度量
        results = {
            'n_hyperedges': len(common_keys),
            'metrics': {}
        }
        
        # 1. L1距离（总变差距离）
        l1_distance = np.sum(np.abs(pred_vals - true_vals))
        results['metrics']['l1_distance'] = float(l1_distance)
        results['metrics']['mean_absolute_error'] = float(np.mean(np.abs(pred_vals - true_vals)))
        
        # 2. L2距离（欧氏距离）
        l2_distance = np.sqrt(np.sum((pred_vals - true_vals) ** 2))
        results['metrics']['l2_distance'] = float(l2_distance)
        results['metrics']['mean_squared_error'] = float(np.mean((pred_vals - true_vals) ** 2))
        results['metrics']['root_mean_squared_error'] = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        
        # 3. KL散度 (需要归一化概率)
        # 为避免log(0)，添加小的平滑项
        epsilon = 1e-10
        pred_normalized = pred_vals / (np.sum(pred_vals) + epsilon)
        true_normalized = true_vals / (np.sum(true_vals) + epsilon)
        
        pred_normalized = np.clip(pred_normalized, epsilon, 1.0)
        true_normalized = np.clip(true_normalized, epsilon, 1.0)
        
        kl_divergence = entropy(true_normalized, pred_normalized)
        results['metrics']['kl_divergence'] = float(kl_divergence)
        
        # 4. JS散度（对称版本的KL散度）
        js_divergence = jensenshannon(true_normalized, pred_normalized) ** 2
        results['metrics']['js_divergence'] = float(js_divergence)
        
        # 5. 最大绝对误差
        max_absolute_error = np.max(np.abs(pred_vals - true_vals))
        results['metrics']['max_absolute_error'] = float(max_absolute_error)
        
        # 6. 相对误差统计
        relative_errors = np.abs(pred_vals - true_vals) / (true_vals + epsilon)
        results['metrics']['mean_relative_error'] = float(np.mean(relative_errors))
        results['metrics']['median_relative_error'] = float(np.median(relative_errors))
        
        # 7. 相关系数
        correlation = np.corrcoef(pred_vals, true_vals)[0, 1]
        results['metrics']['correlation'] = float(correlation)
        
        # 8. 按概率大小分段统计
        # 将超边按真实概率分为高、中、低三档
        sorted_indices = np.argsort(true_vals)
        n = len(true_vals)
        
        low_third = sorted_indices[:n//3]
        mid_third = sorted_indices[n//3:2*n//3]
        high_third = sorted_indices[2*n//3:]
        
        results['metrics']['low_prob_mae'] = float(np.mean(np.abs(pred_vals[low_third] - true_vals[low_third])))
        results['metrics']['mid_prob_mae'] = float(np.mean(np.abs(pred_vals[mid_third] - true_vals[mid_third])))
        results['metrics']['high_prob_mae'] = float(np.mean(np.abs(pred_vals[high_third] - true_vals[high_third])))
        
        # 9. 概率范围统计
        results['statistics'] = {
            'predicted': {
                'min': float(np.min(pred_vals)),
                'max': float(np.max(pred_vals)),
                'mean': float(np.mean(pred_vals)),
                'median': float(np.median(pred_vals)),
                'std': float(np.std(pred_vals))
            },
            'ground_truth': {
                'min': float(np.min(true_vals)),
                'max': float(np.max(true_vals)),
                'mean': float(np.mean(true_vals)),
                'median': float(np.median(true_vals)),
                'std': float(np.std(true_vals))
            }
        }
        
        return results
    
    def compare_multiple_methods(self,
                                 predictions: Dict[str, Dict],
                                 ground_truth_probs: Dict,
                                 **kwargs) -> Dict[str, Any]:
        """
        比较多个方法的预测结果
        
        Args:
            predictions: 方法名到预测概率字典的映射
            ground_truth_probs: 真实概率字典
            **kwargs: 其他参数
            
        Returns:
            比较结果
        """
        comparison = {}
        
        for method_name, pred_probs in predictions.items():
            results = self.evaluate(
                pred_probs,
                ground_truth_probs,
                **kwargs
            )
            comparison[method_name] = results
        
        # 添加排名
        if len(predictions) > 1:
            metrics_to_rank = ['mean_absolute_error', 'mean_squared_error', 
                             'kl_divergence', 'js_divergence']
            
            rankings = {}
            for metric in metrics_to_rank:
                method_scores = {}
                for method_name in predictions.keys():
                    if 'metrics' in comparison[method_name]:
                        score = comparison[method_name]['metrics'].get(metric, float('inf'))
                        method_scores[method_name] = score
                
                # 排序（升序，越小越好）
                sorted_methods = sorted(method_scores.items(), key=lambda x: x[1])
                rankings[metric] = [m[0] for m in sorted_methods]
            
            comparison['rankings'] = rankings
        
        return comparison

