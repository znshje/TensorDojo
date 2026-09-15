# 均方误差 MSE
# pred 与 target 同形，返回所有元素平方误差的均值标量。禁止调用 mse_loss。
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

import torch
import math


def solve(pred, target):
    """均方误差 MSE：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    pred = torch.tensor([-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
     -0.014521554280266614, -0.6123619822872842], dtype=torch.float64).reshape((6,))
    target = torch.tensor([-1.1835467384762002, -1.4830545941372606, 1.8004360883421553, 0.009574053403183548,
     0.15344739309453567, -2.663087663100214], dtype=torch.float64).reshape((6,))
    expected = torch.tensor(1.7449325378505287, dtype=torch.float64).reshape(())
    try:
        actual = solve(pred, target)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
