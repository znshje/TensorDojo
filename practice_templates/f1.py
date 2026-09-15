# F1：二分类调和平均
# scores: [N] 概率，targets: [N] 的 long 类型 0/1 标签，N>0。scores >= threshold 预测为正类（含等号）。返回标量 Tensor；分母为 0 时返回 0。本类题目仅检查数值，无需梯度。
# https://scikit-learn.org/stable/modules/model_evaluation.html

import torch
import math


def solve(scores, targets, threshold=0.5):
    """F1：二分类调和平均：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    scores = torch.tensor([0.9, 0.6, 0.7, 0.2, 0.5], dtype=torch.float64).reshape((5,))
    targets = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int64).reshape((5,))
    expected = torch.tensor(0.5714285714285714, dtype=torch.float64).reshape(())
    try:
        actual = solve(scores, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
