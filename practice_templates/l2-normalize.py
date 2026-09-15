# L2 向量归一化
# x 为浮点 Tensor。沿 dim 计算 L2 范数，分母为 max(norm,eps)，不是 sqrt(sum(x²)+eps)。返回同形 Tensor，全零向量输出 0。禁止 normalize。
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

import torch
import math


def solve(x, dim=-1, eps=1e-12):
    """L2 向量归一化：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    x = torch.tensor([[-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133],
     [-0.014521554280266614, -0.6123619822872842, -1.1835467384762002, -1.4830545941372606],
     [1.8004360883421553, 0.009574053403183548, 0.15344739309453567, -2.663087663100214]], dtype=torch.float64).reshape((3, 4))
    expected = torch.tensor([[-0.9581867051991179, 0.1583726933846457, 0.02307028054921712, 0.2372005272446335],
     [-0.00728317833637185, -0.30712563113668206, -0.5935991285359269, -0.743815082270858],
     [0.5594426988542692, 0.0029749094175199756, 0.04768012832046998, -0.8274911612675706]], dtype=torch.float64).reshape((3, 4))
    try:
        actual = solve(x)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
