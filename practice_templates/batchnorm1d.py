# BatchNorm1d：训练与推理
# x: [N,C] 或 [N,C,L]。weight、bias、running_mean、running_var 均 [C]。训练时沿非通道轴统计：前向使用总体方差 unbiased=False，更新 running_var 使用无偏方差；每通道元素数>1。更新为 (1-momentum)*旧值+momentum*新值。推理只用传入的 running stats。返回 (y,new_running_mean,new_running_var)，后两个停止梯度，禁止原地修改输入。禁止调用现成 BatchNorm。
# https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

import torch
import math


def solve(x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
    """BatchNorm1d：训练与推理：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    x = torch.tensor([[-1.4135131348736631, 0.23363075389153515], [0.03403318414312811, 0.34991725416350133],
     [-0.014521554280266614, -0.6123619822872842], [-1.1835467384762002, -1.4830545941372606]], dtype=torch.float64).reshape((4, 2))
    weight = torch.tensor([1.8004360883421553, 0.009574053403183548], dtype=torch.float64).reshape((2,))
    bias = torch.tensor([0.15344739309453567, -2.663087663100214], dtype=torch.float64).reshape((2,))
    running_mean = torch.tensor([-1.4311441654862587, -0.5483032284032923], dtype=torch.float64).reshape((2,))
    running_var = torch.tensor([1.344624275002011, 0.6533347476510319], dtype=torch.float64).reshape((2,))
    expected = (torch.tensor([[-1.9465628081686919, -2.6551561341531884], [2.0057957706001983, -2.6536480685612167],
     [1.8732226470832043, -2.6661274208761108], [-1.318666037136568, -2.6774190288103394]], dtype=torch.float64).reshape((4, 2)), torch.tensor([-1.3524684550248078, -0.5312696197722008], dtype=torch.float64).reshape((2,)), torch.tensor([1.2681362695319407, 0.6606688361679881], dtype=torch.float64).reshape((2,)))
    try:
        actual = solve(x, weight, bias, running_mean, running_var)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
