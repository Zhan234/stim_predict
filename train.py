"""训练脚本 - 训练各种超图权重预测方法"""

import argparse
import numpy as np
from typing import List, Optional
import stim

from circuits import CircuitFactory
from methods import NoiseCalibrationPredictor, CorrelationPredictor, RLBasedPredictor
from utils import DataManager


def train_predictors(
    code_type: str,
    distance: int,
    rounds: int,
    noise_level: float,
    n_shots: int,
    methods: List[str],
    experiment_name: str,
    num_workers: int = 8,
    rl_epochs: int = 50,
    rl_batch_size: int = 32
):
    """
    训练多个预测方法
    
    Args:
        code_type: 编码类型 ('surface_code', 'repetition_code', 等)
        distance: 码距
        rounds: 测量轮数
        noise_level: 噪声水平
        n_shots: 采样次数
        methods: 要训练的方法列表
        experiment_name: 实验名称
        num_workers: 并行工作线程数
        rl_epochs: RL方法的训练轮数
        rl_batch_size: RL方法的批次大小
    """
    print("=" * 80)
    print(f"开始训练实验: {experiment_name}")
    print(f"编码类型: {code_type}, 码距: {distance}, 轮数: {rounds}, 噪声: {noise_level}")
    print(f"采样次数: {n_shots}")
    print(f"训练方法: {', '.join(methods)}")
    print("=" * 80)
    
    # 1. 生成电路
    print("\n[1/4] 生成电路...")
    circuit = CircuitFactory.create_circuit(
        code_type=code_type,
        distance=distance,
        rounds=rounds,
        noise_level=noise_level
    )
    print(f"电路生成完成，包含 {circuit.num_detectors} 个探测器")
    
    # 2. 采样数据
    print("\n[2/4] 采样探测器数据...")
    sampler = circuit.compile_detector_sampler()
    detector_samples, observables = sampler.sample(shots=n_shots, separate_observables=True)
    print(f"采样完成，形状: detectors={detector_samples.shape}, observables={observables.shape}")
    
    # 3. 保存训练数据
    print("\n[3/4] 保存训练数据...")
    data_manager = DataManager()
    metadata = {
        'code_type': code_type,
        'distance': distance,
        'rounds': rounds,
        'noise_level': noise_level,
        'n_shots': n_shots
    }
    data_manager.save_training_data(
        experiment_name=experiment_name,
        circuit=circuit,
        detector_samples=detector_samples,
        observables=observables,
        metadata=metadata
    )
    print(f"训练数据已保存（包含ground truth DEM用于评测）")
    
    # 4. 训练各个方法
    print("\n[4/4] 训练预测方法...")
    
    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"训练方法: {method_name}")
        print(f"{'='*60}")
        
        try:
            if method_name == 'noise_calibration':
                predictor = NoiseCalibrationPredictor(num_workers=num_workers)
                result = predictor.train(circuit, detector_samples)
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                print(f"负概率数量: {len(result['negative_prob_indices'])}")
                
            elif method_name == 'correlation':
                predictor = CorrelationPredictor(use_numerical=True, num_workers=num_workers)
                result = predictor.train(circuit, detector_samples)
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                
            elif method_name == 'rl_based':
                predictor = RLBasedPredictor(
                    epochs=rl_epochs,
                    batch_size=rl_batch_size,
                    num_workers=num_workers
                )
                result = predictor.train(
                    circuit, 
                    detector_samples,
                    observables=observables
                )
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                print(f"最终平均奖励: {result['final_mean_reward']:.3f}")
                print(f"最终平均LER: {result['final_mean_ler']:.6f}")
                
            else:
                print(f"警告: 未知的方法 '{method_name}'，跳过")
                continue
            
            # 保存预测结果
            data_manager.save_prediction_results(
                experiment_name=experiment_name,
                method_name=method_name,
                hyperedge_probs=result['hyperedge_probs'],
                additional_data={k: v for k, v in result.items() if k != 'hyperedge_probs'}
            )
            print(f"预测结果已保存")
            
        except Exception as e:
            print(f"错误: 训练方法 '{method_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"实验 '{experiment_name}' 训练完成！")
    print("=" * 80)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='训练超图权重预测方法')
    
    parser.add_argument('--code-type', type=str, default='surface_code',
                       choices=['surface_code', 'repetition_code', 'color_code'],
                       help='编码类型')
    parser.add_argument('--distance', type=int, default=5,
                       help='码距')
    parser.add_argument('--rounds', type=int, default=5,
                       help='测量轮数')
    parser.add_argument('--noise', type=float, default=0.001,
                       help='噪声水平')
    parser.add_argument('--shots', type=int, default=100000,
                       help='采样次数')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['noise_calibration', 'correlation'],
                       help='要训练的方法列表')
    parser.add_argument('--experiment', type=str, required=True,
                       help='实验名称')
    parser.add_argument('--workers', type=int, default=8,
                       help='并行工作线程数')
    parser.add_argument('--rl-epochs', type=int, default=50,
                       help='RL方法的训练轮数')
    parser.add_argument('--rl-batch-size', type=int, default=32,
                       help='RL方法的批次大小')
    
    args = parser.parse_args()
    
    train_predictors(
        code_type=args.code_type,
        distance=args.distance,
        rounds=args.rounds,
        noise_level=args.noise,
        n_shots=args.shots,
        methods=args.methods,
        experiment_name=args.experiment,
        num_workers=args.workers,
        rl_epochs=args.rl_epochs,
        rl_batch_size=args.rl_batch_size
    )


if __name__ == '__main__':
    main()

