# LayerNorm：多维 normalized_shape
# normalized_shape 是与 x 尾部若干维度完全匹配的 tuple；weight、bias 形状等于 normalized_shape。沿这些尾部维度联合归一化，使用总体方差。返回与 x 同形 Tensor，禁止 layer_norm/LayerNorm。
# https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

import torch
import math


def solve(x, weight, bias, normalized_shape, eps=1e-5):
    """LayerNorm：多维 normalized_shape：在此实现。"""
    # TODO: 根据题目约定返回结果
    raise NotImplementedError("开始你的实现")


# ===== 本地快速运行：python solution.py =====
if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_printoptions(precision=5, linewidth=100)
    x = torch.tensor([[[1.1334044055743933, 0.4620772467152532, 0.04206122944820574, 0.34594601821099746],
      [-1.5464744767198004, -0.22191597946342112, 0.8037272890414212, 1.082874177318385],
      [0.00892651332797321, -1.0717673967531753, 0.18870746079472087, -0.5132712934449802]],
     [[1.2443654765471568, -1.0055826189462997, -0.25786793006654335, -1.2053463219529492],
      [0.3394419912229643, -1.1676854170730067, 0.28562991374400915, -0.15793455122745947],
      [0.07328307989912629, -1.087081643313971, 0.34598239722946345, -0.7440676020865674]]], dtype=torch.float64).reshape((2, 3, 4))
    weight = torch.tensor([0.23616925010702955, -0.6650426889189348, -0.1146812320816423, -0.30826217861239497], dtype=torch.float64).reshape((4,))
    bias = torch.tensor([1.823971812649229, 0.6361530501124633, 0.7596479991201553, 0.4431789173207746], dtype=torch.float64).reshape((4,))
    normalized_shape = (4,)
    expected = torch.tensor([[[2.2015520537626436, 0.6925149328957234, 0.890160214569001, 0.559078425765928],
      [1.463129253783961, 0.7982833560018634, 0.6735759941425733, 0.12839549839081815],
      [1.9948944867730765, 1.6168502914669565, 0.634709295766966, 0.547536419127952]],
     [[2.204484119381119, 1.119548586441031, 0.7538991623995555, 0.7312344234271485],
      [2.0249480490504053, 1.7277706058091737, 0.6722617808596993, 0.4344093087278887],
      [1.9964802828758723, 1.4727757843121956, 0.6222883674049987, 0.6497754050542044]]], dtype=torch.float64).reshape((2, 3, 4))
    try:
        actual = solve(x, weight, bias, normalized_shape)
    except NotImplementedError:
        print("模板已就绪：请先补全 solve，再运行本文件。")
    else:
        print("实际输出:", actual)
        print("期望输出:", expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, check_dtype=False)
        print("快速样例通过；完整边界与梯度检查请在页面提交判题。")
