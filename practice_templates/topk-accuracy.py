# Top-k Accuracy
# logits: [N,C]，targets: [N] long，1<=k<=C，各行 logits 不含并列值。真实类别在前 k 个预测中即为命中，返回命中率标量 Tensor。不要先 softmax。
# https://scikit-learn.org/stable/modules/model_evaluation.html

import torch
import math


def solve(logits, targets, k=1):
    """Top-k Accuracy：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    logits = torch.tensor([[1.1334044055743933, 0.4620772467152532, 0.04206122944820574, 0.34594601821099746],
     [-1.5945281557811837, -1.3358975990189317, 0.39975937921625415, -0.5182359019396884],
     [0.00892651332797321, -1.0717673967531753, 0.18870746079472087, -0.5132712934449802],
     [-0.41126789034091465, -2.290042350672562, 0.24982736396850957, -0.5754154776340655],
     [0.3394419912229643, -1.1676854170730067, 0.28562991374400915, -0.15793455122745947]], dtype=torch.float64).reshape((5, 4))
    targets = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64).reshape((5,))
    expected = torch.tensor(0.6, dtype=torch.float64).reshape(())
    try:
        actual = solve(logits, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
