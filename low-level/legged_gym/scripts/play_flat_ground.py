# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

def play(args):
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    
    # 强制使用平地环境任务
    if args.task == "b1z1":
        args.task = "b1z1_flat"
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        env_cfg.env.num_envs = 1
    
    # 强制设置为平地地形
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.env.episode_length_s = 60 # make episodes longer for visualization
    
    # 设置命令范围，让机器狗有移动指令
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0] # 前后移动
    env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5] # 左右移动  
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0] # 转向

    if args.observe_gait_commands:
        env_cfg.env.observe_gait_commands = True

    if args.stand_by:
        env_cfg.env.stand_by = True

    # check if observation matches
    from legged_gym.utils.helpers import update_cfg_from_args
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    
        # 设置args为可视化模式（除非明确指定headless）
    if not hasattr(args, 'headless') or not args.headless:
        args.headless = False
    
    # prepare environment 
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    obs = env.get_observations()
    
    # load policy
    print(f"Loading model from: {log_pth}/model_{args.checkpoint}.pt")
    try:
        ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    except:
        # 备用加载方式
        result = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg)
        if isinstance(result, tuple):
            ppo_runner = result[0]
        else:
            ppo_runner = result
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("✅ 成功！低级模型在平地环境中成功加载和运行！")
    total_params = sum(p.numel() for p in ppo_runner.alg.actor_critic.parameters())
    print(f"模型参数总数: {total_params}")
    print(f"环境信息: {env.cfg.env.num_envs} 个环境，观测维度: {env.cfg.env.num_observations}")
    
    # 重置环境，就像原始play.py一样
    env.reset()
    
    # 手动设置locomotion commands确保腿部运动
    print("🦿 手动设置locomotion commands确保腿部运动...")
    env.commands[:, 0] = 0.5  # 前进速度 0.5 m/s
    env.commands[:, 1] = 0.0  # 横向速度
    env.commands[:, 2] = 0.0  # 转向速度  
    print(f"✅ 设置commands: x_vel={env.commands[0, 0]:.2f}, y_vel={env.commands[0, 1]:.2f}, yaw_vel={env.commands[0, 2]:.2f}")
    
    # 运行长时间仿真，参考原始play.py
    traj_length = 1000 * int(env.max_episode_length)
    print(f"🎮 开始运行 {traj_length} 步仿真...")
    print("🎯 机器狗将根据环境命令自动移动")
    print("⏹️  按 Ctrl+C 停止仿真")
    
    import time
    try:
        for i in range(traj_length):
            start_time = time.time()
            
            # 使用与原始play.py相同的策略调用方式
            actions = policy(obs.detach(), hist_encoding=True)
            
            # 执行步骤，匹配原始play.py的返回值格式
            obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
            
            # 每100步重新设置commands，防止被环境重置为0
            if i % 100 == 0:
                env.commands[:, 0] = 0.5  # 保持前进命令
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0
                if i % 1000 == 0:
                    print(f"步数: {i:6d}, 总奖励: {rews.item():.3f}, 手臂奖励: {arm_rews.item():.3f}, commands已重设")
            elif i % 1000 == 0:
                print(f"步数: {i:6d}, 总奖励: {rews.item():.3f}, 手臂奖励: {arm_rews.item():.3f}")
            
            # 控制执行频率，保持实时仿真
            stop_time = time.time()
            duration = stop_time - start_time
            time.sleep(max(0.02 - duration, 0))  # 50Hz频率
            
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断，在第 {i} 步停止")
    except Exception as e:
        print(f"\n❌ 运行异常: {e}")
    
    print("✅ 仿真结束！")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)