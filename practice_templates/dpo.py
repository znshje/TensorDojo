# DPO：偏好优化损失
# 四个输入均为 [B] 的序列 log probability 之和。chosen/rejected 属于当前策略，ref_* 属于参考策略且须停止梯度。返回 batch 平均损失，需支持较大的 log probability 差。
# https://arxiv.org/abs/2305.18290

import torch
import math


def solve(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    """DPO：偏好优化损失：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    chosen = torch.tensor([-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
     -0.014521554280266614], dtype=torch.float64).reshape((5,))
    rejected = torch.tensor([-0.6123619822872842, -1.1835467384762002, -1.4830545941372606, 1.8004360883421553,
     0.009574053403183548], dtype=torch.float64).reshape((5,))
    ref_chosen = torch.tensor([0.15344739309453567, -2.663087663100214, -1.4311441654862587, -0.5483032284032923,
     0.3231804016508869], dtype=torch.float64).reshape((5,))
    ref_rejected = torch.tensor([-0.47796881088083604, 1.561822332787565, -0.12975446077824895, -0.13349894401057621,
     1.2739817330799412], dtype=torch.float64).reshape((5,))
    expected = torch.tensor(0.6347869253629902, dtype=torch.float64).reshape(())
    try:
        actual = solve(chosen, rejected, ref_chosen, ref_rejected)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
