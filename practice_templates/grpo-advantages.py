# GRPO：组内优势归一化
# rewards: [B,G]，每行是同一 prompt 的 G 个回答奖励。本题约定总体标准差 unbiased=False，分母为 std + eps。返回同形优势并停止梯度。G=1 或组内奖励相同应返回 0。
# https://arxiv.org/abs/2402.03300

import torch
import math


def solve(rewards, eps=1e-6):
    """GRPO：组内优势归一化：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    rewards = torch.tensor([[-1.4135131348736631, 0.23363075389153515, 0.03403318414312811, 0.34991725416350133,
      -0.014521554280266614],
     [-0.6123619822872842, -1.1835467384762002, -1.4830545941372606, 1.8004360883421553,
      0.009574053403183548],
     [0.15344739309453567, -2.663087663100214, -1.4311441654862587, -0.5483032284032923,
      0.3231804016508869]], dtype=torch.float64).reshape((3, 5))
    expected = torch.tensor([[-1.9567557459296205, 0.6187600649815439, 0.3066642607152617, 0.8005885755863439,
      0.23074284464647077],
     [-0.2734678246840796, -0.7637838996132614, -1.0208872529625865, 1.7977248441840794,
      0.26041413307584854],
     [0.8934061675743159, -1.657005572118696, -0.5414635688183651, 0.25796120143817447,
      1.0471017719245712]], dtype=torch.float64).reshape((3, 5))
    try:
        actual = solve(rewards)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
