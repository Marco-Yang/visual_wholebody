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
    
    # 强制使用平地任务，但加载原始b1z1的模型
    original_task = args.task
    args.task = "b1z1_flat"  # 使用平地环境
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing - 使用与原始play.py相同的设置
    env_cfg.env.num_envs = 1
    
    # 使用与原始play.py相同的terrain设置，但我们已经修改了manip_loco.py使用平地
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = False

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
    
    # prepare environment - 使用平地环境但加载原始模型
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy - 需要使用原始任务名来加载模型
    args.task = original_task  # 恢复原始任务名用于加载模型
    
    # load policy - 使用与原始play.py完全相同的方式
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, checkpoint, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("✅ 成功！低级模型在平地环境中成功加载和运行！")
    total_params = sum(p.numel() for p in ppo_runner.alg.actor_critic.parameters())
    print(f"模型参数总数: {total_params}")
    print(f"环境信息: {env.cfg.env.num_envs} 个环境，观测维度: {env.cfg.env.num_observations}")
    
    # 使用与原始play.py完全相同的相机设置
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    
    # 使用与原始play.py完全相同的仿真长度
    traj_length = 1000*int(env.max_episode_length)
    
    print(f"🎮 开始运行 {traj_length} 步仿真...")
    print("🎯 机器狗将根据环境命令自动移动")
    print("⏹️  按 Ctrl+C 停止仿真")
    
    # 重置环境 - 与原始play.py相同
    env.reset()
    
    # 主循环 - 与原始play.py完全相同的逻辑
    import time
    try:
        for i in range(traj_length):
            start_time = time.time()
            
            # 使用与原始play.py完全相同的策略调用
            actions = policy(obs.detach(), hist_encoding=True)
            obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
            
            if not args.headless:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if args.headless:
                # 在headless模式下每1000步输出一次状态
                if i % 1000 == 0:
                    print(f"步数: {i:6d}, 总奖励: {rews.item():.3f}, 手臂奖励: {arm_rews.item():.3f}")
            
            # 与原始play.py相同的时间控制
            stop_time = time.time()
            duration = stop_time - start_time
            time.sleep(max(0.02 - duration, 0))
            
    except KeyboardInterrupt:
        print(f"🛑 用户中断，在第 {i} 步停止")
    
    print("✅ 仿真结束！")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)