# 稳定的 BCE with logits
# logits 与 targets 同形，targets 在 [0,1]。返回所有元素的二元交叉熵均值；需支持 ±1000 的 logits。禁止 binary_cross_entropy_with_logits 和 binary_cross_entropy。
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

import torch
import math


def solve(logits, targets):
    """稳定的 BCE with logits：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    logits = torch.tensor([[-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133],
     [-0.014521554280266614, -0.6123619822872842, -1.1835467384762002, -1.4830545941372606],
     [1.8004360883421553, 0.009574053403183548, 0.15344739309453567, -2.663087663100214]], dtype=torch.float64).reshape((3, 4))
    targets = torch.tensor([[0.5582304164472548, 0.6909972264829735, 0.8446242750020111, 0.15333474765103194],
     [0.9868078799550212, 0.707140588585892, 0.2666170014639203, 0.5597558026904579],
     [0.7422668827208556, 0.9702360747677321, 0.10516025851116384, 0.2590600610206225]], dtype=torch.float64).reshape((3, 4))
    expected = torch.tensor(0.7647269539547294, dtype=torch.float64).reshape(())
    try:
        actual = solve(logits, targets)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
