# IoU：分割交并比
# pred、target 为同形 bool 张量。所有维度合并统计一个整体指标，返回标量 Tensor。本题约定双空掩码得分为 1；仅一侧为空得分为 0。
# https://scikit-learn.org/stable/modules/model_evaluation.html

import torch
import math


def solve(pred, target):
    """IoU：分割交并比：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    pred = torch.tensor([[True, True], [False, False]], dtype=torch.bool).reshape((2, 2))
    target = torch.tensor([[True, False], [True, False]], dtype=torch.bool).reshape((2, 2))
    expected = torch.tensor(0.3333333333333333, dtype=torch.float64).reshape(())
    try:
        actual = solve(pred, target)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
