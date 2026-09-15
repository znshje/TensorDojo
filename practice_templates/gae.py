# PPO：广义优势估计 GAE
# rewards、dones: [T,B]；values: [T+1,B]，最后一行用于 bootstrap。dones[t] 表示这一步后环境真正终止（不是时间截断）。反向递推，终止处切断。返回 (advantages, returns)，均 [T,B] 且不带梯度。
# https://arxiv.org/abs/1506.02438

import torch
import math


def solve(rewards, values, dones, gamma=0.99, lam=0.95):
    """PPO：广义优势估计 GAE：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    rewards = torch.tensor([[-1.4135131348736631, 0.23363075389153515], [0.03403318414312811, 0.34991725416350133],
     [-0.014521554280266614, -0.6123619822872842], [-1.1835467384762002, -1.4830545941372606]], dtype=torch.float64).reshape((4, 2))
    values = torch.tensor([[1.8004360883421553, 0.009574053403183548], [0.15344739309453567, -2.663087663100214],
     [-1.4311441654862587, -0.5483032284032923], [0.3231804016508869, -0.47796881088083604],
     [1.561822332787565, -0.12975446077824895]], dtype=torch.float64).reshape((5, 2))
    dones = torch.tensor([[False, False], [True, False], [False, True], [False, False]], dtype=torch.bool).reshape((4, 2))
    expected = (torch.tensor([[-3.1743453675710267, -0.14585390191973424], [-0.11941420895140756, 2.4099374631165613],
     [1.7736992984976825, -0.06405875388399196], [0.03947696933260236, -1.1335426994268911]], dtype=torch.float64).reshape((4, 2)), torch.tensor([[-1.3739092792288714, -0.1362798485165507], [0.034033184143128115, -0.2531501999836525],
     [0.3425551330114238, -0.6123619822872842], [0.36265737098348927, -1.611511510307727]], dtype=torch.float64).reshape((4, 2)))
    try:
        actual = solve(rewards, values, dones)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
