# 数值稳定的 Softmax
# 输入 x 为任意浮点张量，沿 dim 归一化，输出与 x 同形。禁止调用 torch.softmax、Tensor.softmax 或 log_softmax。需正确处理绝对值很大的 logits。
# https://arxiv.org/abs/1706.03762

import torch
import math


def solve(x, dim=-1):
    """数值稳定的 Softmax：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    x = torch.tensor([[-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
      -0.014521554280266614],
     [-0.6123619822872842, -1.1835467384762002, -1.4830545941372606, 1.8004360883421553,
      0.009574053403183548],
     [0.15344739309453567, -2.663087663100214, -1.4311441654862587, -0.5483032284032923,
      0.3231804016508869]], dtype=torch.float64).reshape((3, 5))
    expected = torch.tensor([[0.049192457634449155, 0.2554136021490573, 0.20919914195954994, 0.28691060909357996,
      0.19928418916336363],
     [0.06661689867398683, 0.03762894330463308, 0.02788992931889844, 0.7437882245070875,
      0.12407600419539405],
     [0.3394952271292765, 0.02030617104418461, 0.06960722224116932, 0.16829346431017,
      0.40229791527519937]], dtype=torch.float64).reshape((3, 5))
    try:
        actual = solve(x)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
