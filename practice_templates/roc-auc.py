# ROC-AUC：排序与并列分数
# scores: [N] 任意实数预测分数，targets: [N] long 0/1。保证同时存在正负样本。返回 ROC 曲线面积标量 Tensor：正样本得分高于负样本记 1，相等记 0.5。不能二值化分数。允许 O(N²) 成对比较，进阶可实现平均秩 O(N log N)。
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

import torch
import math


def solve(scores, targets):
    """ROC-AUC：排序与并列分数：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float64).reshape((4,))
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64).reshape((4,))
    expected = torch.tensor(0.75, dtype=torch.float64).reshape(())
    try:
        actual = solve(scores, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
