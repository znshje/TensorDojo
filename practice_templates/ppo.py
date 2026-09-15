# PPO：裁剪策略损失
# new_logp、old_logp、advantages: [N]。计算最小化形式的 clipped policy loss；old_logp 与 advantages 必须 detach。仅实现策略项，不含 critic/entropy。clip_eps 在 (0,1)。
# https://arxiv.org/abs/1707.06347

import torch
import math


def solve(new_logp, old_logp, advantages, clip_eps=0.2):
    """PPO：裁剪策略损失：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    new_logp = torch.tensor([0.0, 0.5, -0.7, 0.8, -0.8], dtype=torch.float64).reshape((5,))
    old_logp = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64).reshape((5,))
    advantages = torch.tensor([1.0, 2.0, 3.0, -2.0, -3.0], dtype=torch.float64).reshape((5,))
    expected = torch.tensor(0.3922651891221415, dtype=torch.float64).reshape(())
    try:
        actual = solve(new_logp, old_logp, advantages)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
