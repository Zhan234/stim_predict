"""评测脚本 - 评测各种预测方法的性能"""

import argparse
import numpy as np
from typing import List, Optional
import stim

from utils import DataManager
from evaluators import DistributionDistanceEvaluator, DecoderLEREvaluator


def evaluate_predictions(
    experiment_name: str,
    methods: List[str],
    evaluators: List[str],
    ground_truth_method: str = 'dem',
    decoders: Optional[List[str]] = None,
    test_shots: int = 100000
):
    """
    评测多个预测方法的性能
    
    Args:
        experiment_name: 实验名称
        methods: 要评测的方法列表
        evaluators: 要使用的评测器列表
        ground_truth_method: 真实值的来源 ('dem' 或某个方法名)
        decoders: 要测试的解码器列表（用于decoder_ler评测器）
        test_shots: 测试集采样次数
    """
    print("=" * 80)
    print(f"开始评测实验: {experiment_name}")
    print(f"评测方法: {', '.join(methods)}")
    print(f"评测器: {', '.join(evaluators)}")
    print(f"测试集大小: {test_shots} shots")
    print("=" * 80)
    
    # 1. 加载元数据和电路
    print("\n[1/4] 加载实验配置...")
    data_manager = DataManager()
    
    try:
        training_data = data_manager.load_training_data(experiment_name)
        circuit = training_data['circuit']
        metadata = training_data['metadata']
        
        print(f"实验配置加载完成")
        print(f"元数据: {metadata}")
    except Exception as e:
        print(f"错误: 无法加载实验配置: {e}")
        return
    
    # 2. 采样新的测试数据（独立于训练数据）
    print(f"\n[2/4] 采样测试数据 ({test_shots} shots)...")
    print("注意: 测试数据独立于训练数据，确保评测结果的泛化性")
    
    sampler = circuit.compile_detector_sampler()
    test_detector_samples, test_observables = sampler.sample(
        shots=test_shots, 
        separate_observables=True
    )
    print(f"测试数据采样完成，形状: detectors={test_detector_samples.shape}, observables={test_observables.shape}")
    
    # 3. 加载各方法的预测结果
    print("\n[3/4] 加载预测结果...")
    predictions = {}
    
    for method_name in methods:
        try:
            pred_data = data_manager.load_prediction_results(experiment_name, method_name)
            predictions[method_name] = pred_data['hyperedge_probs']
            print(f"  - {method_name}: {len(pred_data['hyperedge_probs'])} 个超边")
        except Exception as e:
            print(f"  警告: 无法加载方法 '{method_name}' 的预测结果: {e}")
    
    if not predictions:
        print("错误: 没有成功加载任何预测结果")
        return
    
    # 获取真实值（从保存的ground truth DEM）
    print(f"\n获取真实值 (方法: {ground_truth_method})...")
    if ground_truth_method == 'dem':
        # 从保存的ground truth DEM文件加载
        import correlation
        import os
        
        dem_path = os.path.join(data_manager.base_dir, experiment_name, "ground_truth.dem")
        if os.path.exists(dem_path):
            with open(dem_path, 'r') as f:
                dem_origin = stim.DetectorErrorModel(f.read())
            print(f"从保存的ground truth DEM加载")
        else:
            # 向后兼容：如果没有保存的DEM，从circuit重新生成
            print("警告: 未找到保存的ground truth DEM，从电路重新生成")
            dem_origin = circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True
            )
        
        tanner_graph = correlation.TannerGraph(dem_origin)
        ground_truth_probs = dict(tanner_graph.hyperedge_probs)
        print(f"获取了 {len(ground_truth_probs)} 个超边的真实概率")
    else:
        # 使用某个方法的预测作为真实值
        if ground_truth_method in predictions:
            ground_truth_probs = predictions[ground_truth_method]
            print(f"使用方法 '{ground_truth_method}' 作为真实值")
        else:
            print(f"错误: 指定的真实值方法 '{ground_truth_method}' 未找到")
            return
    
    # 4. 运行评测
    print("\n[4/4] 运行评测...")
    
    for evaluator_name in evaluators:
        print(f"\n{'='*60}")
        print(f"评测器: {evaluator_name}")
        print(f"{'='*60}")
        
        try:
            if evaluator_name == 'distribution_distance':
                evaluator = DistributionDistanceEvaluator()
                results = evaluator.compare_multiple_methods(
                    predictions=predictions,
                    ground_truth_probs=ground_truth_probs
                )
                
                # 打印结果
                print("\n概率分布距离评测结果:")
                for method_name, result in results.items():
                    if method_name == 'rankings':
                        continue
                    
                    print(f"\n  方法: {method_name}")
                    if 'error' in result:
                        print(f"    错误: {result['error']}")
                    elif 'metrics' in result:
                        metrics = result['metrics']
                        print(f"    MAE: {metrics.get('mean_absolute_error', 'N/A'):.6f}")
                        print(f"    RMSE: {metrics.get('root_mean_squared_error', 'N/A'):.6f}")
                        print(f"    KL散度: {metrics.get('kl_divergence', 'N/A'):.6f}")
                        print(f"    JS散度: {metrics.get('js_divergence', 'N/A'):.6f}")
                        print(f"    相关系数: {metrics.get('correlation', 'N/A'):.6f}")
                
                # 打印排名
                if 'rankings' in results:
                    print("\n  排名 (越小越好):")
                    for metric, ranking in results['rankings'].items():
                        print(f"    {metric}: {' > '.join(ranking)}")
                
                # 保存结果
                data_manager.save_evaluation_results(
                    experiment_name=experiment_name,
                    evaluator_name='distribution_distance',
                    results=results
                )
                
            elif evaluator_name == 'decoder_ler':
                if decoders is None:
                    decoders = ['pymatching', 'pymatching_corr']
                
                evaluator = DecoderLEREvaluator(decoders=decoders)
                results = evaluator.compare_multiple_methods(
                    predictions=predictions,
                    ground_truth_probs=ground_truth_probs,
                    circuit=circuit,
                    detector_samples=test_detector_samples,
                    observables=test_observables
                )
                
                # 打印结果
                print("\n解码器逻辑错误率评测结果:")
                for method_name, result in results.items():
                    if method_name == 'rankings':
                        continue
                    
                    print(f"\n  方法: {method_name}")
                    if 'decoders' in result:
                        for decoder_name, decoder_result in result['decoders'].items():
                            if 'error' in decoder_result:
                                print(f"    {decoder_name}: 错误 - {decoder_result['error']}")
                            else:
                                pred_ler = decoder_result['predicted_dem']['ler']
                                true_ler = decoder_result['ground_truth_dem']['ler']
                                ratio = decoder_result['ler_ratio']
                                print(f"    {decoder_name}:")
                                print(f"      预测DEM LER: {pred_ler:.6f} ({decoder_result['predicted_dem']['n_errors']}/{result['n_shots']})")
                                print(f"      真实DEM LER: {true_ler:.6f} ({decoder_result['ground_truth_dem']['n_errors']}/{result['n_shots']})")
                                print(f"      LER比率: {ratio:.3f}")
                
                # 打印排名
                if 'rankings' in results:
                    print("\n  排名 (按LER，越小越好):")
                    for decoder_name, ranking in results['rankings'].items():
                        print(f"    {decoder_name}:")
                        for item in ranking:
                            print(f"      {item['method']}: LER={item['ler']:.6f}")
                
                # 保存结果
                data_manager.save_evaluation_results(
                    experiment_name=experiment_name,
                    evaluator_name='decoder_ler',
                    results=results
                )
            
            else:
                print(f"警告: 未知的评测器 '{evaluator_name}'，跳过")
            
        except Exception as e:
            print(f"错误: 评测器 '{evaluator_name}' 运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"实验 '{experiment_name}' 评测完成！")
    print("=" * 80)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='评测超图权重预测方法')
    
    parser.add_argument('--experiment', type=str, required=True,
                       help='实验名称')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['noise_calibration', 'correlation'],
                       help='要评测的方法列表')
    parser.add_argument('--evaluators', type=str, nargs='+',
                       default=['distribution_distance', 'decoder_ler'],
                       choices=['distribution_distance', 'decoder_ler'],
                       help='要使用的评测器列表')
    parser.add_argument('--ground-truth', type=str, default='dem',
                       help='真实值来源 (dem 或某个方法名)')
    parser.add_argument('--decoders', type=str, nargs='+',
                       default=['pymatching', 'pymatching_corr'],
                       help='要测试的解码器列表')
    parser.add_argument('--test-shots', type=int, default=100000,
                       help='测试集采样次数（默认100000）')
    
    args = parser.parse_args()
    
    evaluate_predictions(
        experiment_name=args.experiment,
        methods=args.methods,
        evaluators=args.evaluators,
        ground_truth_method=args.ground_truth,
        decoders=args.decoders,
        test_shots=args.test_shots
    )


if __name__ == '__main__':
    main()

