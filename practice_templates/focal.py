# 二分类 Focal Loss
# logits 与 targets 同形；targets 只含 0/1。正类权重 alpha，负类权重 1-alpha，gamma >= 0。所有元素取均值，需数值稳定。
# https://arxiv.org/abs/1708.02002

import torch
import math


def solve(logits, targets, gamma=2.0, alpha=0.25):
    """二分类 Focal Loss：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    logits = torch.tensor([-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
     -0.014521554280266614, -0.6123619822872842], dtype=torch.float64).reshape((6,))
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64).reshape((6,))
    expected = torch.tensor(0.046635886790347865, dtype=torch.float64).reshape(())
    try:
        actual = solve(logits, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
