#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import isaacgym

import torch
import numpy as np
import os
import argparse
import time

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for flat ground visualization  
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 1000 * 24 * env_cfg.sim.dt  # 1000 episodes * 24 steps/episode  
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.measure_heights = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 使用平地任务但确保有locomotion commands
    if args.task == 'b1z1':
        args.task = 'b1z1_flat'
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 1
    
    # 确保正确的模型加载路径
    args.task = 'b1z1'  # 加载b1z1模型
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    args.task = 'b1z1_flat'  # 但使用平地环境

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print(f"观测维度: {obs.shape}")
    print(f"动作维度: {env.num_actions}")
    print(f"模型参数数量: {sum(p.numel() for p in policy.parameters())}")
    
    # 手动设置locomotion commands来确保腿部运动
    print("手动设置locomotion commands...")
    env.commands[:, 0] = 0.5  # 前进速度
    env.commands[:, 1] = 0.0  # 横向速度  
    env.commands[:, 2] = 0.0  # 转向速度
    print(f"设置commands: {env.commands[0]}")
    
    obs = env.reset()
    episode_length = 0
    
    if not args.headless:
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - camera_position
        img_idx = 0

    for i in range(1000 * int(env.max_episode_length)):
        actions = policy(obs.detach(), hist_encoding=True)
        obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
        
        # 每100步重新设置commands确保不会被重置为0
        if i % 100 == 0:
            env.commands[:, 0] = 0.5  # 保持前进命令
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            print(f"步骤 {i}: 重新设置commands = {env.commands[0]}")
        
        episode_length += 1

        if not args.headless:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if args.headless:
            # 在headless模式下每50步输出一次状态
            if i % 50 == 0:
                base_pos = env.root_states[0, :3]
                base_vel = env.base_lin_vel[0]
                print(f"步骤 {i}: 位置={base_pos.cpu().numpy()}, 速度={base_vel.cpu().numpy()}")
        
        time.sleep(0.02)  # 50Hz
        
        if episode_length > 1000:
            break

if __name__ == '__main__':
    args = get_args()
    play(args)