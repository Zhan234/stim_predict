"""训练脚本 - 训练各种超图权重预测方法"""

import argparse
import json
import os
import numpy as np
from typing import List, Optional, Dict, Any
import stim

from circuits import CircuitFactory
from methods import CorrelationPredictor, RLBasedPredictor, GRPOPredictor, ACPredictor, DQNPredictor
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
    correlation_use_numerical: bool = True,
    correlation_num_workers: Optional[int] = None,
    skip_existing: bool = True,
    rl_epochs: int = 50,
    rl_batch_size: int = 32,
    rl_learning_rate: float = 1e-3,
    rl_clip_ratio: float = 0.2,
    rl_entropy_coef: float = 0.01,
    rl_value_coef: float = 0.5,
    rl_max_grad_norm: float = 0.5,
    rl_use_gpu: bool = True,
    grpo_epochs: int = 100,
    grpo_group_size: int = 64,
    grpo_learning_rate: float = 1e-6,
    grpo_clip_ratio: float = 0.2,
    grpo_kl_coef: float = 0.04,
    grpo_max_grad_norm: float = 0.5,
    grpo_supervision_mode: str = 'outcome',
    grpo_use_gpu: bool = True,
    ac_epochs: int = 100,
    ac_batch_size: int = 32,
    ac_learning_rate_actor: float = 1e-4,
    ac_learning_rate_critic: float = 1e-3,
    ac_hidden_dim: int = 128,
    ac_gamma: float = 0.99,
    ac_entropy_coef: float = 0.01,
    ac_max_grad_norm: float = 0.5,
    ac_use_gpu: bool = True,
    ac_eval_frequency: int = 10,
    ac_action_scale: float = 0.05
    dqn_epochs: int = 50,
    dqn_batch_size: int = 32,
    dqn_lr_actor: float = 1e-3,
    dqn_lr_critic: float = 1e-3,
    dqn_buffer_size: int = 1000,
    dqn_exploration_noise: float = 2.0,
    dqn_use_gpu: bool = True
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
        
        # 如果存在同名预测结果且选择跳过，直接跳过以避免重复计算
        pred_path = os.path.join(data_manager.base_dir, experiment_name, "predictions", f"{method_name}.pkl")
        if skip_existing and os.path.exists(pred_path):
            print(f"检测到已有结果，跳过训练: {method_name}")
            continue
        
        try:
            if method_name == 'correlation':
                predictor = CorrelationPredictor(
                    use_numerical=correlation_use_numerical,
                    num_workers=correlation_num_workers or num_workers
                )
                result = predictor.train(circuit, detector_samples)
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                
            elif method_name == 'rl_based':
                predictor = RLBasedPredictor(
                    epochs=rl_epochs,
                    batch_size=rl_batch_size,
                    learning_rate=rl_learning_rate,
                    clip_ratio=rl_clip_ratio,
                    entropy_coef=rl_entropy_coef,
                    value_coef=rl_value_coef,
                    max_grad_norm=rl_max_grad_norm,
                    use_gpu=rl_use_gpu
                )
                result = predictor.train(
                    circuit, 
                    detector_samples,
                    observables=observables
                )
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                print(f"最终平均奖励: {result['final_mean_reward']:.3f}")
                print(f"最终平均LER: {result['final_mean_ler']:.6f}")
                
            elif method_name == 'grpo':
                predictor = GRPOPredictor(
                    epochs=grpo_epochs,
                    group_size=grpo_group_size,
                    learning_rate=grpo_learning_rate,
                    clip_ratio=grpo_clip_ratio,
                    kl_coef=grpo_kl_coef,
                    max_grad_norm=grpo_max_grad_norm,
                    supervision_mode=grpo_supervision_mode,
                    use_gpu=grpo_use_gpu
                )
                result = predictor.train(
                    circuit, 
                    detector_samples,
                    observables=observables
                )
                print(f"训练完成，共 {len(result['hyperedge_probs'])} 个超边")
                print(f"最终平均奖励: {result['final_mean_reward']:.3f}")
                print(f"最终平均LER: {result['final_mean_ler']:.6f}")
                
            elif method_name == 'actor_critic' or method_name == 'ac':
                predictor = ACPredictor(
                    learning_rate_actor=ac_learning_rate_actor,
                    learning_rate_critic=ac_learning_rate_critic,
                    batch_size=ac_batch_size,
                    epochs=ac_epochs,
                    hidden_dim=ac_hidden_dim,
                    gamma=ac_gamma,
                    entropy_coef=ac_entropy_coef,
                    max_grad_norm=ac_max_grad_norm,
                    use_gpu=ac_use_gpu,
                    eval_frequency=ac_eval_frequency,
                    action_scale=ac_action_scale
                )
                result = predictor.train(
                    circuit, 
            elif method_name == 'dqn':
                predictor = DQNPredictor(
                    epochs=dqn_epochs,
                    batch_size=dqn_batch_size,
                    learning_rate_actor=dqn_lr_actor,
                    learning_rate_critic=dqn_lr_critic,
                    buffer_size=dqn_buffer_size,
                    exploration_noise=dqn_exploration_noise,
                    use_gpu=dqn_use_gpu
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


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def merge_config_and_args(config: Optional[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """
    合并配置文件和命令行参数
    优先级：命令行参数 > 配置文件
    
    Args:
        config: 配置字典（可为None）
        args: 命令行参数
        
    Returns:
        合并后的参数字典
    """
    params = {}
    
    if config:
        # 从配置文件读取参数
        params['experiment_name'] = config.get('experiment_name', '')
        params['code_type'] = config.get('circuit', {}).get('code_type', 'surface_code')
        params['distance'] = config.get('circuit', {}).get('distance', 5)
        params['rounds'] = config.get('circuit', {}).get('rounds', 5)
        params['noise_level'] = config.get('circuit', {}).get('noise_level', 0.001)
        params['n_shots'] = config.get('sampling', {}).get('n_shots', 100000)
        params['num_workers'] = config.get('training', {}).get('num_workers', 8)
        params['methods'] = config.get('training', {}).get('methods', ['correlation'])
        params['correlation_use_numerical'] = config.get('training', {}).get('correlation', {}).get('use_numerical', True)
        params['correlation_num_workers'] = config.get('training', {}).get('correlation', {}).get('num_workers', None)
        params['skip_existing'] = config.get('training', {}).get('skip_existing', True)
        params['rl_epochs'] = config.get('training', {}).get('rl_based', {}).get('epochs', 50)
        params['rl_batch_size'] = config.get('training', {}).get('rl_based', {}).get('batch_size', 32)
        params['rl_learning_rate'] = config.get('training', {}).get('rl_based', {}).get('learning_rate', 1e-3)
        params['rl_clip_ratio'] = config.get('training', {}).get('rl_based', {}).get('clip_ratio', 0.2)
        params['rl_entropy_coef'] = config.get('training', {}).get('rl_based', {}).get('entropy_coef', 0.01)
        params['rl_value_coef'] = config.get('training', {}).get('rl_based', {}).get('value_coef', 0.5)
        params['rl_max_grad_norm'] = config.get('training', {}).get('rl_based', {}).get('max_grad_norm', 0.5)
        params['rl_use_gpu'] = config.get('training', {}).get('rl_based', {}).get('use_gpu', True)
        params['grpo_epochs'] = config.get('training', {}).get('grpo', {}).get('epochs', 100)
        params['grpo_group_size'] = config.get('training', {}).get('grpo', {}).get('group_size', 64)
        params['grpo_learning_rate'] = config.get('training', {}).get('grpo', {}).get('learning_rate', 1e-6)
        params['grpo_clip_ratio'] = config.get('training', {}).get('grpo', {}).get('clip_ratio', 0.2)
        params['grpo_kl_coef'] = config.get('training', {}).get('grpo', {}).get('kl_coef', 0.04)
        params['grpo_max_grad_norm'] = config.get('training', {}).get('grpo', {}).get('max_grad_norm', 0.5)
        params['grpo_supervision_mode'] = config.get('training', {}).get('grpo', {}).get('supervision_mode', 'outcome')
        params['grpo_use_gpu'] = config.get('training', {}).get('grpo', {}).get('use_gpu', True)
        params['ac_epochs'] = config.get('training', {}).get('ac', {}).get('epochs', 100)
        params['ac_batch_size'] = config.get('training', {}).get('ac', {}).get('batch_size', 32)
        params['ac_learning_rate_actor'] = config.get('training', {}).get('ac', {}).get('learning_rate_actor', 1e-4)
        params['ac_learning_rate_critic'] = config.get('training', {}).get('ac', {}).get('learning_rate_critic', 1e-3)
        params['ac_hidden_dim'] = config.get('training', {}).get('ac', {}).get('hidden_dim', 128)
        params['ac_gamma'] = config.get('training', {}).get('ac', {}).get('gamma', 0.99)
        params['ac_entropy_coef'] = config.get('training', {}).get('ac', {}).get('entropy_coef', 0.01)
        params['ac_max_grad_norm'] = config.get('training', {}).get('ac', {}).get('max_grad_norm', 0.5)
        params['ac_use_gpu'] = config.get('training', {}).get('ac', {}).get('use_gpu', True)
        params['ac_eval_frequency'] = config.get('training', {}).get('ac', {}).get('eval_frequency', 10)
        params['ac_action_scale'] = config.get('training', {}).get('ac', {}).get('action_scale', 0.05)
    else:
        # 使用默认值
        params['experiment_name'] = ''
        params['code_type'] = 'surface_code'
        params['distance'] = 5
        params['rounds'] = 5
        params['noise_level'] = 0.001
        params['n_shots'] = 100000
        params['num_workers'] = 8
        params['methods'] = ['correlation']
        params['correlation_use_numerical'] = True
        params['correlation_num_workers'] = None
        params['skip_existing'] = True
        params['rl_epochs'] = 50
        params['rl_batch_size'] = 32
        params['rl_learning_rate'] = 1e-3
        params['rl_clip_ratio'] = 0.2
        params['rl_entropy_coef'] = 0.01
        params['rl_value_coef'] = 0.5
        params['rl_max_grad_norm'] = 0.5
        params['rl_use_gpu'] = True
        params['grpo_epochs'] = 100
        params['grpo_group_size'] = 64
        params['grpo_learning_rate'] = 1e-6
        params['grpo_clip_ratio'] = 0.2
        params['grpo_kl_coef'] = 0.04
        params['grpo_max_grad_norm'] = 0.5
        params['grpo_supervision_mode'] = 'outcome'
        params['grpo_use_gpu'] = True
        params['ac_epochs'] = 100
        params['ac_batch_size'] = 32
        params['ac_learning_rate_actor'] = 1e-4
        params['ac_learning_rate_critic'] = 1e-3
        params['ac_hidden_dim'] = 128
        params['ac_gamma'] = 0.99
        params['ac_entropy_coef'] = 0.01
        params['ac_max_grad_norm'] = 0.5
        params['ac_use_gpu'] = True
        params['ac_eval_frequency'] = 10
        params['ac_action_scale'] = 0.05
    
    # 命令行参数覆盖配置文件（优先级更高）
    if args.experiment is not None:
        params['experiment_name'] = args.experiment
    if args.code_type is not None:
        params['code_type'] = args.code_type
    if args.distance is not None:
        params['distance'] = args.distance
    if args.rounds is not None:
        params['rounds'] = args.rounds
    if args.noise is not None:
        params['noise_level'] = args.noise
    if args.shots is not None:
        params['n_shots'] = args.shots
    if args.workers is not None:
        params['num_workers'] = args.workers
    if args.methods is not None:
        params['methods'] = args.methods
    if getattr(args, 'correlation_use_numerical', None):
        params['correlation_use_numerical'] = True
    if getattr(args, 'correlation_no_numerical', None):
        params['correlation_use_numerical'] = False
    if getattr(args, 'correlation_num_workers', None) is not None:
        params['correlation_num_workers'] = args.correlation_num_workers
    if getattr(args, 'skip_existing', None) is not None:
        params['skip_existing'] = args.skip_existing
    if args.rl_epochs is not None:
        params['rl_epochs'] = args.rl_epochs
    if args.rl_batch_size is not None:
        params['rl_batch_size'] = args.rl_batch_size
    if getattr(args, 'rl_learning_rate', None) is not None:
        params['rl_learning_rate'] = args.rl_learning_rate
    if getattr(args, 'rl_clip_ratio', None) is not None:
        params['rl_clip_ratio'] = args.rl_clip_ratio
    if getattr(args, 'rl_entropy_coef', None) is not None:
        params['rl_entropy_coef'] = args.rl_entropy_coef
    if getattr(args, 'rl_value_coef', None) is not None:
        params['rl_value_coef'] = args.rl_value_coef
    if getattr(args, 'rl_max_grad_norm', None) is not None:
        params['rl_max_grad_norm'] = args.rl_max_grad_norm
    if getattr(args, 'rl_use_gpu', None) is not None:
        params['rl_use_gpu'] = args.rl_use_gpu
    if args.grpo_epochs is not None:
        params['grpo_epochs'] = args.grpo_epochs
    if args.grpo_group_size is not None:
        params['grpo_group_size'] = args.grpo_group_size
    if getattr(args, 'grpo_learning_rate', None) is not None:
        params['grpo_learning_rate'] = args.grpo_learning_rate
    if getattr(args, 'grpo_clip_ratio', None) is not None:
        params['grpo_clip_ratio'] = args.grpo_clip_ratio
    if getattr(args, 'grpo_kl_coef', None) is not None:
        params['grpo_kl_coef'] = args.grpo_kl_coef
    if getattr(args, 'grpo_max_grad_norm', None) is not None:
        params['grpo_max_grad_norm'] = args.grpo_max_grad_norm
    if getattr(args, 'grpo_supervision_mode', None) is not None:
        params['grpo_supervision_mode'] = args.grpo_supervision_mode
    if getattr(args, 'grpo_use_gpu', None) is not None:
        params['grpo_use_gpu'] = args.grpo_use_gpu
    if getattr(args, 'ac_epochs', None) is not None:
        params['ac_epochs'] = args.ac_epochs
    if getattr(args, 'ac_batch_size', None) is not None:
        params['ac_batch_size'] = args.ac_batch_size
    if getattr(args, 'ac_learning_rate_actor', None) is not None:
        params['ac_learning_rate_actor'] = args.ac_learning_rate_actor
    if getattr(args, 'ac_learning_rate_critic', None) is not None:
        params['ac_learning_rate_critic'] = args.ac_learning_rate_critic
    if getattr(args, 'ac_hidden_dim', None) is not None:
        params['ac_hidden_dim'] = args.ac_hidden_dim
    if getattr(args, 'ac_gamma', None) is not None:
        params['ac_gamma'] = args.ac_gamma
    if getattr(args, 'ac_entropy_coef', None) is not None:
        params['ac_entropy_coef'] = args.ac_entropy_coef
    if getattr(args, 'ac_max_grad_norm', None) is not None:
        params['ac_max_grad_norm'] = args.ac_max_grad_norm
    if getattr(args, 'ac_use_gpu', None) is not None:
        params['ac_use_gpu'] = args.ac_use_gpu
    if getattr(args, 'ac_eval_frequency', None) is not None:
        params['ac_eval_frequency'] = args.ac_eval_frequency
    if getattr(args, 'ac_action_scale', None) is not None:
        params['ac_action_scale'] = args.ac_action_scale
    
    return params


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='训练超图权重预测方法')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（JSON格式）')
    parser.add_argument('--code-type', type=str, default=None,
                       choices=['surface_code', 'repetition_code', 'color_code'],
                       help='编码类型（覆盖配置文件）')
    parser.add_argument('--distance', type=int, default=None,
                       help='码距（覆盖配置文件）')
    parser.add_argument('--rounds', type=int, default=None,
                       help='测量轮数（覆盖配置文件）')
    parser.add_argument('--noise', type=float, default=None,
                       help='噪声水平（覆盖配置文件）')
    parser.add_argument('--shots', type=int, default=None,
                       help='采样次数（覆盖配置文件）')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='要训练的方法列表（覆盖配置文件）')
    parser.add_argument('--experiment', type=str, default=None,
                       help='实验名称（必需，或从配置文件读取）')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作线程数（覆盖配置文件）')
    parser.add_argument('--skip-existing', action='store_true',
                       help='如已有同名预测结果则跳过训练')
    parser.add_argument('--correlation-use-numerical', action='store_true',
                       help='强制使用数值相关性方法（覆盖配置文件）')
    parser.add_argument('--correlation-no-numerical', action='store_true',
                       help='强制关闭数值相关性（覆盖配置文件）')
    parser.add_argument('--correlation-num-workers', type=int, default=None,
                       help='Correlation方法的线程数（覆盖配置文件）')
    parser.add_argument('--rl-epochs', type=int, default=None,
                       help='RL方法的训练轮数（覆盖配置文件）')
    parser.add_argument('--rl-batch-size', type=int, default=None,
                       help='RL方法的批次大小（覆盖配置文件）')
    parser.add_argument('--grpo-epochs', type=int, default=None,
                       help='GRPO方法的训练轮数（覆盖配置文件）')
    parser.add_argument('--grpo-group-size', type=int, default=None,
                       help='GRPO方法的组大小（覆盖配置文件）')
    parser.add_argument('--ac-epochs', type=int, default=None,
                       help='Actor-Critic方法的训练轮数（覆盖配置文件）')
    parser.add_argument('--ac-batch-size', type=int, default=None,
                       help='Actor-Critic方法的批次大小（覆盖配置文件）')
    
    args = parser.parse_args()
    
    # 加载配置文件（如果提供）
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            print(f"已加载配置文件: {args.config}")
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}")
            print("将使用命令行参数和默认值")
    
    # 合并配置和命令行参数
    params = merge_config_and_args(config, args)
    
    # 检查必需参数
    if not params['experiment_name']:
        parser.error("--experiment 参数是必需的，或必须在配置文件中指定 experiment_name")
    
    # 打印使用的参数
    print("\n使用的训练参数:")
    print(f"  实验名称: {params['experiment_name']}")
    print(f"  编码类型: {params['code_type']}")
    print(f"  码距: {params['distance']}")
    print(f"  轮数: {params['rounds']}")
    print(f"  噪声水平: {params['noise_level']}")
    print(f"  采样次数: {params['n_shots']}")
    print(f"  方法列表: {params['methods']}")
    print(f"  工作线程数: {params['num_workers']}")
    print(f"  已有结果跳过训练: {params['skip_existing']}")
    if 'rl_based' in params['methods']:
        print(f"  RL训练轮数: {params['rl_epochs']}")
        print(f"  RL批次大小: {params['rl_batch_size']}")
        print(f"  RL学习率: {params['rl_learning_rate']}")
        print(f"  RL裁剪阈值: {params['rl_clip_ratio']}")
        print(f"  RL熵系数: {params['rl_entropy_coef']}")
        print(f"  RL价值系数: {params['rl_value_coef']}")
        print(f"  RL梯度裁剪: {params['rl_max_grad_norm']}")
        print(f"  RL使用GPU: {params['rl_use_gpu']}")
    if 'grpo' in params['methods']:
        print(f"  GRPO训练轮数: {params['grpo_epochs']}")
        print(f"  GRPO组大小: {params['grpo_group_size']}")
        print(f"  GRPO学习率: {params['grpo_learning_rate']}")
        print(f"  GRPO裁剪阈值: {params['grpo_clip_ratio']}")
        print(f"  GRPO KL系数: {params['grpo_kl_coef']}")
        print(f"  GRPO梯度裁剪: {params['grpo_max_grad_norm']}")
        print(f"  GRPO监督模式: {params['grpo_supervision_mode']}")
        print(f"  GRPO使用GPU: {params['grpo_use_gpu']}")
    if 'ac' in params['methods'] or 'actor_critic' in params['methods']:
        print(f"  AC训练轮数: {params['ac_epochs']}")
        print(f"  AC批次大小: {params['ac_batch_size']}")
        print(f"  AC Actor学习率: {params['ac_learning_rate_actor']}")
        print(f"  AC Critic学习率: {params['ac_learning_rate_critic']}")
        print(f"  AC隐藏层维度: {params['ac_hidden_dim']}")
        print(f"  AC折扣因子: {params['ac_gamma']}")
        print(f"  AC熵系数: {params['ac_entropy_coef']}")
        print(f"  AC梯度裁剪: {params['ac_max_grad_norm']}")
        print(f"  AC使用GPU: {params['ac_use_gpu']}")
        print(f"  AC评估频率: 每{params['ac_eval_frequency']}轮")
        print(f"  AC动作缩放: {params['ac_action_scale']}")
    print()
    
    train_predictors(
        code_type=params['code_type'],
        distance=params['distance'],
        rounds=params['rounds'],
        noise_level=params['noise_level'],
        n_shots=params['n_shots'],
        methods=params['methods'],
        experiment_name=params['experiment_name'],
        num_workers=params['num_workers'],
        correlation_use_numerical=params['correlation_use_numerical'],
        correlation_num_workers=params['correlation_num_workers'],
        rl_epochs=params['rl_epochs'],
        rl_batch_size=params['rl_batch_size'],
        rl_learning_rate=params['rl_learning_rate'],
        rl_clip_ratio=params['rl_clip_ratio'],
        rl_entropy_coef=params['rl_entropy_coef'],
        rl_value_coef=params['rl_value_coef'],
        rl_max_grad_norm=params['rl_max_grad_norm'],
        rl_use_gpu=params['rl_use_gpu'],
        grpo_epochs=params['grpo_epochs'],
        grpo_group_size=params['grpo_group_size'],
        grpo_learning_rate=params['grpo_learning_rate'],
        grpo_clip_ratio=params['grpo_clip_ratio'],
        grpo_kl_coef=params['grpo_kl_coef'],
        grpo_max_grad_norm=params['grpo_max_grad_norm'],
        grpo_supervision_mode=params['grpo_supervision_mode'],
        grpo_use_gpu=params['grpo_use_gpu'],
        ac_epochs=params['ac_epochs'],
        ac_batch_size=params['ac_batch_size'],
        ac_learning_rate_actor=params['ac_learning_rate_actor'],
        ac_learning_rate_critic=params['ac_learning_rate_critic'],
        ac_hidden_dim=params['ac_hidden_dim'],
        ac_gamma=params['ac_gamma'],
        ac_entropy_coef=params['ac_entropy_coef'],
        ac_max_grad_norm=params['ac_max_grad_norm'],
        ac_use_gpu=params['ac_use_gpu'],
        ac_eval_frequency=params['ac_eval_frequency'],
        ac_action_scale=params['ac_action_scale']
        dqn_epochs=params.get('dqn_epochs', 200),
        dqn_batch_size=params.get('dqn_batch_size', 32),
        dqn_lr_actor=params.get('dqn_lr_actor', 1e-3),
        dqn_lr_critic=params.get('dqn_lr_critic', 1e-3),
        dqn_buffer_size=params.get('dqn_buffer_size', 1000),
        dqn_exploration_noise=params.get('dqn_exploration_noise', 2.0),
        dqn_use_gpu=params.get('dqn_use_gpu', True)
    )


if __name__ == '__main__':
    main()
