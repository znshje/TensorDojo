# Macro-F1：多分类宏平均
# pred、targets: [N] long 标签，num_classes=C。分别计算全部 C 个类别的 F1，再等权平均。某类分母为 0 时该类 F1=0，仍参与平均。返回标量 Tensor；不是先平均 Precision/Recall 再计算 F1。
# https://scikit-learn.org/stable/modules/model_evaluation.html

import torch
import math


def solve(pred, targets, num_classes):
    """Macro-F1：多分类宏平均：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    pred = torch.tensor([0, 1, 1, 2, 0], dtype=torch.int64).reshape((5,))
    targets = torch.tensor([0, 2, 1, 2, 1], dtype=torch.int64).reshape((5,))
    num_classes = 4
    expected = torch.tensor(0.45833333333333326, dtype=torch.float64).reshape(())
    try:
        actual = solve(pred, targets, num_classes)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
