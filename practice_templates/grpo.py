# GRPO：带 KL 的 token 策略损失
# new_logp、old_logp、ref_logp、mask: [B,G,T]；advantages: [B,G] 为已算好的组内优势。mask 为 bool，True 表示有效 token。先对每条回答有效 token 求均值，再对非空回答求均值。全空返回可求导 0。old/ref/advantages 停止梯度。使用下述 k3 KL 估计。本题不实现采样、奖励模型与优化器。
# https://arxiv.org/abs/2402.03300

import torch
import math


def solve(new_logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.04):
    """GRPO：带 KL 的 token 策略损失：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    new_logp = torch.tensor([[[-0.28270262697473264, 0.04672615077830703, 0.006806636828625622, 0.06998345083270027],
      [-0.002904310856053323, -0.12247239645745685, -0.23670934769524005, -0.2966109188274521],
      [0.3600872176684311, 0.0019148106806367096, 0.030689478618907137, -0.5326175326200427]]], dtype=torch.float64).reshape((1, 3, 4))
    old_logp = torch.tensor([[[-0.2862288330972517, -0.10966064568065846, 0.06463608033017738, -0.09559376217616722],
      [0.31236446655751304, -0.02595089215564979, -0.026699788802115244, 0.25479634661598827],
      [-0.025753588425386664, -0.5296162829830237, 0.1222811204097005, 0.09504832702237119]]], dtype=torch.float64).reshape((1, 3, 4))
    ref_logp = torch.tensor([[[0.29756193235243494, 0.10928805906505311, 0.10060965136622235, 0.03759425192514506],
      [-0.28678611923232084, -0.07396916875259006, 0.2402453714518286, 0.15013998664175773],
      [0.00743240537923924, 0.28262664142328275, 0.13883940337367032, 0.21014901394402397]]], dtype=torch.float64).reshape((1, 3, 4))
    advantages = torch.tensor([[1.4099929894411634, 0.08303720317694699, -0.7958497036472395]], dtype=torch.float64).reshape((1, 3))
    mask = torch.tensor([[[True, True, True, True], [True, False, False, False], [False, False, False, False]]], dtype=torch.bool).reshape((1, 3, 4))
    expected = torch.tensor(-0.7857792711053674, dtype=torch.float64).reshape(())
    try:
        actual = solve(new_logp, old_logp, ref_logp, advantages, mask)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
