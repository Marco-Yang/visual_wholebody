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

from .manip_loco.quadruped_manipulator_env import QuadrupedManipulatorEnvironment
from .manip_loco.robot_system_config import QuadrupedManipulatorConfig, QuadrupedManipulatorPPOConfig
from .manip_loco.quadruped_manipulator_flat_env import QuadrupedManipulatorFlatEnvironment  
from .manip_loco.robot_system_flat_config import QuadrupedManipulatorFlatConfig, QuadrupedManipulatorFlatPPOConfig

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "quadruped_manipulator", QuadrupedManipulatorEnvironment, QuadrupedManipulatorConfig(), QuadrupedManipulatorPPOConfig(), 'b1z1')
task_registry.register( "quadruped_manipulator_flat", QuadrupedManipulatorFlatEnvironment, QuadrupedManipulatorFlatConfig(), QuadrupedManipulatorFlatPPOConfig(), 'b1z1_flat')

# 为了兼容性，也注册b1z1任务名称
task_registry.register( "b1z1", QuadrupedManipulatorEnvironment, QuadrupedManipulatorConfig(), QuadrupedManipulatorPPOConfig(), 'b1z1')
task_registry.register( "b1z1_flat", QuadrupedManipulatorFlatEnvironment, QuadrupedManipulatorFlatConfig(), QuadrupedManipulatorFlatPPOConfig(), 'b1z1_flat')