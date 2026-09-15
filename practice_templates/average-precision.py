# Average Precision：PR 曲线
# scores: [N] 分数，targets: [N] long 0/1。按不同分数阈值降序，将并列分数一起纳入，计算 AP=Σ(Recall增量×当前Precision)。无正样本返回 0。这是非插值 AP，不是对 PR 曲线做梯形积分。
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

import torch
import math


def solve(scores, targets):
    """Average Precision：PR 曲线：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float64).reshape((4,))
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64).reshape((4,))
    expected = torch.tensor(0.8333333333333333, dtype=torch.float64).reshape(())
    try:
        actual = solve(scores, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
