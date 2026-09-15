# 知识蒸馏 KL 散度
# student_logits、teacher_logits: [B,C]。计算 KL(teacher || student)，对类别求和、batch 求均值，再乘 temperature²。teacher 必须停止梯度。temperature > 0。禁止 kl_div。
# https://arxiv.org/abs/1503.02531

import torch
import math


def solve(student_logits, teacher_logits, temperature=1.0):
    """知识蒸馏 KL 散度：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    student_logits = torch.tensor([[-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
      -0.014521554280266614],
     [-0.6123619822872842, -1.1835467384762002, -1.4830545941372606, 1.8004360883421553,
      0.009574053403183548],
     [0.15344739309453567, -2.663087663100214, -1.4311441654862587, -0.5483032284032923,
      0.3231804016508869]], dtype=torch.float64).reshape((3, 5))
    teacher_logits = torch.tensor([[-0.47796881088083604, 1.561822332787565, -0.12975446077824895, -0.13349894401057621,
      1.2739817330799412],
     [-0.12876794212693332, -2.648081414915118, 0.6114056020485025, 0.4752416351118559,
      1.4878096617621746],
     [0.5464402953252655, 0.5030482568311118, 0.18797125962572528, -1.4339305961616042,
      -0.3698458437629503]], dtype=torch.float64).reshape((3, 5))
    expected = torch.tensor(0.6559659108770235, dtype=torch.float64).reshape(())
    try:
        actual = solve(student_logits, teacher_logits)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
