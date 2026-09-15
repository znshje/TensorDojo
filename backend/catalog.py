"""Executable exercise definitions. Shapes and algorithm variants are explicit."""
import inspect
import textwrap
import torch

PROBLEMS = {}

def add(id, title, category, difficulty, minutes, description, signature, formula, hints, reference, cases, source='https://arxiv.org/abs/1706.03762', forbidden=()):
    solution = textwrap.dedent(inspect.getsource(reference))
    solution = solution.replace('def '+reference.__name__+'(', 'def solve(', 1)
    PROBLEMS[id] = dict(id=id, title=title, category=category, difficulty=difficulty, minutes=minutes,
        description=description, signature=signature, formula=formula, hints=hints, source=source,
        starter='import torch\nimport math\n\n\ndef solve('+signature+'):\n    """'+title+'：在此实现。"""\n    # TODO: 根据题目约定返回结果\n    raise NotImplementedError("开始你的实现")\n',
        solution='import torch\nimport math\n\n'+solution, reference=reference, cases=cases, forbidden=forbidden)

def rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)

def softmax(x, dim=-1):
    z=x-x.max(dim=dim, keepdim=True).values
    return z.exp()/z.exp().sum(dim=dim, keepdim=True)

add('softmax','数值稳定的 Softmax','基础算子','入门',15,
    '输入 x 为任意浮点张量，沿 dim 归一化，输出与 x 同形。禁止调用 torch.softmax、Tensor.softmax 或 log_softmax。需正确处理绝对值很大的 logits。',
    'x, dim=-1','softmax(xᵢ) = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))',
    ['先减去目标维度的最大值。','所有归约保留维度，避免广播错位。'],softmax,
    lambda:[('二维 / 最后一维',(rand(3,5),)),('极大 logits',(rand(2,6)*1000,)),('指定维度',(rand(2,3,4),1)),('单元素',(rand(1,1),))],forbidden=('softmax','log_softmax'))

def self_attn(x,wq,wk,wv,mask=None):
    q,k,v=x@wq,x@wk,x@wv
    s=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if mask is not None: s=s.masked_fill(~mask,float('-inf'))
    p=torch.softmax(s,dim=-1)
    p=torch.nan_to_num(p,nan=0.0)
    return p@v

def attention_cases(cross=False):
    out=[]
    for label,b,t,s,d,dv in [('标准形状',2,4,4,6,3),('不同投影宽度',1,3,5,4,7),('单 token',2,1,1,2,2)]:
        if cross: args=(rand(b,t,d),rand(b,s,d),rand(d,4),rand(d,4),rand(d,dv))
        else: args=(rand(b,t,d),rand(d,4),rand(d,4),rand(d,dv)); s=t
        out.append((label,args))
        mask=torch.ones(t,s,dtype=torch.bool).tril(); mask[0]=False
        out.append((label+' / mask 含全遮挡行',args+(mask,)))
    return out

add('self-attention','自注意力 Self-Attention','注意力机制','基础',25,
    'x: [B,T,D]；wq,wk: [D,Dk]；wv: [D,Dv]。返回 [B,T,Dv]。mask 可广播至 [B,T,T]，True 表示可见；全遮挡查询的输出必须为 0。不含偏置、dropout 和输出投影。',
    'x, wq, wk, wv, mask=None','Q = XWq, K = XWk, V = XWv\nAttention = softmax(QKᵀ / √Dk + mask)V',
    ['缩放因子使用每个 key 的维度。','全遮挡行的 softmax 会产生 NaN，需显式置零。'],self_attn,attention_cases)

def cross_attn(x,context,wq,wk,wv,mask=None):
    q,k,v=x@wq,context@wk,context@wv
    s=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if mask is not None: s=s.masked_fill(~mask,float('-inf'))
    p=torch.nan_to_num(torch.softmax(s,dim=-1),nan=0.0)
    return p@v

add('cross-attention','交叉注意力 Cross-Attention','注意力机制','基础',25,
    'x: [B,T,D]，context: [B,S,D]。Q 来自 x，K/V 来自 context；wq,wk: [D,Dk]，wv: [D,Dv]。输出 [B,T,Dv]。mask 广播至 [B,T,S]，True 可见，全遮挡行输出 0。',
    'x, context, wq, wk, wv, mask=None','Q = XWq, K = CWk, V = CWv',
    ['查询长度 T 与上下文长度 S 不必相等。','注意转置只作用于最后两个维度。'],cross_attn,lambda:attention_cases(True))

def mha(x,wq,wk,wv,wo,num_heads,mask=None):
    b,t,d=x.shape
    h=num_heads
    q=(x@wq).reshape(b,t,h,-1).transpose(1,2)
    k=(x@wk).reshape(b,t,h,-1).transpose(1,2)
    v=(x@wv).reshape(b,t,h,-1).transpose(1,2)
    s=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if mask is not None: s=s.masked_fill(~mask,float('-inf'))
    p=torch.nan_to_num(torch.softmax(s,dim=-1),nan=0.0)
    return (p@v).transpose(1,2).reshape(b,t,-1)@wo

def mha_cases():
    cases=[]
    for h in [1,2,4]:
        args=(rand(2,4,8),rand(8,8),rand(8,8),rand(8,8),rand(8,5),h)
        cases.extend([(f'{h} 个头',args),(f'{h} 个头 / 因果 mask',args+(torch.ones(4,4,dtype=torch.bool).tril(),))])
    return cases
add('mha','多头注意力 MHA','注意力机制','进阶',35,
    'x: [B,T,D]；wq,wk,wv: [D,D]；wo: [D,Dout]；D 能被 num_heads 整除。拆头后为 [B,H,T,D/H]，拼接后应用 wo。输出 [B,T,Dout]。mask 广播至 [B,H,T,T]，True 可见，全遮挡输出为 0。',
    'x, wq, wk, wv, wo, num_heads, mask=None','headᵢ = Attention(Qᵢ,Kᵢ,Vᵢ)\nY = Concat(head₁,…,headₕ)Wo',
    ['reshape 后交换序列轴与 head 轴。','合头前先转回 [B,T,H,Dh]。'],mha,mha_cases)

def grouped(q,k,v,mask=None):
    repeats=q.shape[1]//k.shape[1]
    k=k.repeat_interleave(repeats,dim=1)
    v=v.repeat_interleave(repeats,dim=1)
    s=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if mask is not None: s=s.masked_fill(~mask,float('-inf'))
    return torch.nan_to_num(torch.softmax(s,dim=-1),nan=0.0)@v

def grouped_cases(mqa=False):
    out=[]
    for h,hkv,t,s in [(4,1 if mqa else 2,3,5),(6,1 if mqa else 3,1,7),(2,1 if mqa else 2,4,4)]:
        args=(rand(2,h,t,4),rand(2,hkv,s,4),rand(2,hkv,s,3))
        out.append((f'Hq={h}, Hkv={hkv}, T={t}, S={s}',args))
        mask=torch.ones(t,s,dtype=torch.bool).tril(); mask[0]=False
        out.append(('含全遮挡行',args+(mask,)))
    return out
for id,title,mqa_flag in [('gqa','分组查询注意力 GQA',False),('mqa','多查询注意力 MQA',True)]:
    add(id,title,'注意力机制','进阶',30,
        'q: [B,Hq,T,Dk]；k: [B,Hkv,S,Dk]；v: [B,Hkv,S,Dv]。Hq 能被 Hkv 整除；连续的 Hq/Hkv 个 query heads 共享一个 KV head。'+('本题 Hkv 固定为 1。' if mqa_flag else '')+'输出 [B,Hq,T,Dv]。mask 广播至 [B,Hq,T,S]，True 可见；全遮挡行输出 0。',
        'q, k, v, mask=None','head i 使用 KV head floor(i / (Hq/Hkv))',
        ['连续分组应使用 repeat_interleave，而不是 repeat。','缩放使用 Dk，输出宽度是 Dv。'],grouped,lambda m=mqa_flag:grouped_cases(m),source='https://arxiv.org/abs/2305.13245')

def mla(q,c,wk,wv,mask=None):
    k=torch.einsum('bsr,hrd->bhsd',c,wk)
    v=torch.einsum('bsr,hrv->bhsv',c,wv)
    scores=q@k.transpose(-1,-2)/q.shape[-1]**0.5
    if mask is not None: scores=scores.masked_fill(~mask,float('-inf'))
    return torch.nan_to_num(torch.softmax(scores,dim=-1),nan=0.0)@v

add('mla','MLA：低秩 KV 压缩核心','注意力机制','挑战',40,
    '这是 MLA 内容注意力的教学子问题，不含 RoPE。q: [B,H,T,Dk]；共享潜变量 c: [B,S,R]；wk: [H,R,Dk]；wv: [H,R,Dv]。从 c 恢复各头 K/V 后计算注意力；输出 [B,H,T,Dv]。mask True 可见，全遮挡行输出 0。允许使用矩阵吸收得到等价输出。',
    'q, c, wk, wv, mask=None','Kʰ = C Wkʰ, Vʰ = C Wvʰ\n缓存 C，而非各头完整 K、V',
    ['用 einsum 同时展开 batch 和 head。','矩阵吸收：先让 Q 乘 Wk 的转置。'],mla,
    lambda:[('低秩 R=2',(rand(2,3,4,5),rand(2,6,2),rand(3,2,5),rand(3,2,4))),('单 token 解码',(rand(1,2,1,4),rand(1,8,3),rand(2,3,4),rand(2,3,6))),('因果 mask',(rand(1,2,3,4),rand(1,3,2),rand(2,2,4),rand(2,2,5),torch.ones(3,3,dtype=torch.bool).tril()))],source='https://arxiv.org/abs/2405.04434')

def mla_rope(qc,qr,c,wk,wv,kr,mask=None):
    k=torch.einsum('bsr,hrd->bhsd',c,wk)
    v=torch.einsum('bsr,hrv->bhsv',c,wv)
    s=(qc@k.transpose(-1,-2)+qr@kr.unsqueeze(1).transpose(-1,-2))/(qc.shape[-1]+qr.shape[-1])**0.5
    if mask is not None: s=s.masked_fill(~mask,float('-inf'))
    return torch.nan_to_num(torch.softmax(s,dim=-1),nan=0.0)@v
add('mla-rope','MLA：解耦 RoPE 注意力','注意力机制','挑战',45,
    'qc: [B,H,T,Dc]；qr: [B,H,T,Dr]；c: [B,S,R]；wk: [H,R,Dc]；wv: [H,R,Dv]；kr: [B,S,Dr]。qr、kr 已应用 RoPE，无需再次旋转。kr 跨头共享。将内容分数与位置分数相加，按 sqrt(Dc+Dr) 缩放。返回 [B,H,T,Dv]。mask True 可见，全遮挡输出 0。不含输入/输出投影。',
    'qc, qr, c, wk, wv, kr, mask=None','S = (Qc Kcᵀ + Qr Krᵀ) / √(Dc + Dr)',
    ['位置 key 在 head 维添加长度为 1 的轴。','位置分支不参与 value 的构造。'],mla_rope,
    lambda:[('预填充',(rand(2,3,4,5),rand(2,3,4,2),rand(2,4,3),rand(3,3,5),rand(3,3,6),rand(2,4,2))),('解码',(rand(1,2,1,3),rand(1,2,1,4),rand(1,7,2),rand(2,2,3),rand(2,2,5),rand(1,7,4)))],source='https://arxiv.org/abs/2405.04434')

def ce(logits,targets,ignore_index=-100):
    logp=logits-torch.logsumexp(logits,dim=-1,keepdim=True)
    valid=targets!=ignore_index
    safe=targets.masked_fill(~valid,0)
    losses=-logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1)
    return (losses*valid).sum()/valid.sum().clamp_min(1)
add('cross-entropy','交叉熵与 ignore_index','损失函数','基础',25,
    'logits: [...,C]，targets: [...] 为 long 类型类别索引。忽略 ignore_index 后取有效位置均值；全部忽略时返回可反向传播的标量 0。禁止调用 cross_entropy、nll_loss。',
    'logits, targets, ignore_index=-100','L = mean_valid(logsumexp(z) − z_target)',
    ['不要先 softmax 再 log，直接使用 logsumexp。','gather 前替换被忽略的 target，防止越界。'],ce,
    lambda:[('分类',(rand(4,5),torch.tensor([0,1,3,4]))),('语言模型 padding',(rand(2,3,5)*100,torch.tensor([[0,1,-100],[4,-100,2]]))),('全部忽略',(rand(2,3),torch.tensor([-100,-100])))],source='https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',forbidden=('cross_entropy','nll_loss'))

def bce(logits,targets):
    return (torch.clamp(logits,min=0)-logits*targets+torch.log1p(torch.exp(-logits.abs()))).mean()
add('bce','稳定的 BCE with logits','损失函数','基础',20,
    'logits 与 targets 同形，targets 在 [0,1]。返回所有元素的二元交叉熵均值；需支持 ±1000 的 logits。禁止 binary_cross_entropy_with_logits 和 binary_cross_entropy。',
    'logits, targets','L = mean(max(z,0) − zy + log(1 + exp(−|z|)))',
    ['采用稳定的 softplus 恒等式。','支持软标签，不只支持 0/1。'],bce,
    lambda:[('软标签',(rand(3,4),torch.rand(3,4,dtype=torch.float64))),('极值',(torch.tensor([-1000.,1000.,-20.,20.],dtype=torch.float64),torch.tensor([1.,0.,0.,1.],dtype=torch.float64)))],source='https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html',forbidden=('binary_cross_entropy_with_logits','binary_cross_entropy'))

def mse(pred,target):
    return ((pred-target)**2).mean()
add('mse','均方误差 MSE','损失函数','入门',10,'pred 与 target 同形，返回所有元素平方误差的均值标量。禁止调用 mse_loss。','pred, target','L = mean((prediction − target)²)', ['返回标量张量，不能用 item() 截断梯度。'],mse,lambda:[('向量',(rand(6),rand(6))),('三维张量',(rand(2,3,4),rand(2,3,4))),('相同输入',(torch.ones(3,dtype=torch.float64),torch.ones(3,dtype=torch.float64)))],source='https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html',forbidden=('mse_loss',))

def kl(student_logits,teacher_logits,temperature=1.0):
    ls=torch.log_softmax(student_logits/temperature,dim=-1)
    lt=torch.log_softmax(teacher_logits.detach()/temperature,dim=-1)
    return (lt.exp()*(lt-ls)).sum(dim=-1).mean()*temperature**2
add('kl','知识蒸馏 KL 散度','损失函数','进阶',25,
    'student_logits、teacher_logits: [B,C]。计算 KL(teacher || student)，对类别求和、batch 求均值，再乘 temperature²。teacher 必须停止梯度。temperature > 0。禁止 kl_div。',
    'student_logits, teacher_logits, temperature=1.0','L = T² · mean_B Σ_C p_teacher(log p_teacher − log p_student)',
    ['注意 KL 的方向。','teacher 概率与 log 概率均从 detach 后的张量计算。'],kl,
    lambda:[('T=1',(rand(3,5),rand(3,5))),('T=3',(rand(2,7),rand(2,7),3.0))],source='https://arxiv.org/abs/1503.02531',forbidden=('kl_div',))

def focal(logits,targets,gamma=2.0,alpha=0.25):
    p=torch.sigmoid(logits)
    pt=torch.where(targets==1,p,1-p)
    at=torch.where(targets==1,alpha,1-alpha)
    loss=torch.clamp(logits,min=0)-logits*targets+torch.log1p(torch.exp(-logits.abs()))
    return (at*(1-pt)**gamma*loss).mean()
add('focal','二分类 Focal Loss','损失函数','进阶',25,
    'logits 与 targets 同形；targets 只含 0/1。正类权重 alpha，负类权重 1-alpha，gamma >= 0。所有元素取均值，需数值稳定。',
    'logits, targets, gamma=2.0, alpha=0.25','L = mean(−αt (1−pt)^γ log(pt))',
    ['用稳定 BCE 项代替直接 log(pt)。','alpha_t 对正负类不同。'],focal,
    lambda:[('标准',(rand(6),torch.tensor([0.,1.,0.,1.,1.,0.],dtype=torch.float64))),('gamma=0',(rand(4),torch.tensor([1.,0.,1.,0.],dtype=torch.float64),0.0,0.7)),('极值',(torch.tensor([-1000.,1000.],dtype=torch.float64),torch.tensor([1.,0.],dtype=torch.float64)))],source='https://arxiv.org/abs/1708.02002')

def rms(x,weight,eps=1e-6):
    return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+eps)*weight
add('rmsnorm','RMSNorm','基础算子','基础',15,'x: [...,D]，weight: [D]。沿最后一维做 RMS 归一化，再乘 weight。eps 在平方根内部。无中心化、无偏置。禁止 rms_norm。','x, weight, eps=1e-6','y = x / √(mean(x²) + ε) · weight',['RMSNorm 不需要减均值。'],rms,lambda:[('三维',(rand(2,3,5),rand(5))),('全零',(torch.zeros(2,4,dtype=torch.float64),rand(4))),('较大 eps',(rand(3,4),rand(4),0.1))],source='https://arxiv.org/abs/1910.07467',forbidden=('rms_norm',))

def layernorm(x,weight,bias,eps=1e-5):
    centered=x-x.mean(dim=-1,keepdim=True)
    return centered*torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+eps)*weight+bias
add('layernorm','LayerNorm','基础算子','基础',20,'x: [...,D]，weight、bias: [D]。最后一维归一化，方差使用总体方差（unbiased=False）。禁止 layer_norm。','x, weight, bias, eps=1e-5','y = (x − μ) / √(σ² + ε) · γ + β',['torch.var 默认可能使用无偏估计，应明确约定。'],layernorm,lambda:[('一般输入',(rand(2,3,5),rand(5),rand(5))),('常量行',(torch.ones(2,4,dtype=torch.float64),rand(4),rand(4))),('D=1',(rand(3,1),rand(1),rand(1)))],source='https://arxiv.org/abs/1607.06450',forbidden=('layer_norm',))

def rope(x,positions,base=10000.0):
    d=x.shape[-1]
    freq=base**(-torch.arange(0,d,2,dtype=x.dtype,device=x.device)/d)
    angles=positions.to(x.dtype)[:,None]*freq
    co,si=angles.cos(),angles.sin()
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*co-odd*si,even*si+odd*co),dim=-1).flatten(-2)
add('rope','旋转位置编码 RoPE','基础算子','进阶',30,
    'x: [B,H,T,D]，D 为偶数；positions: [T] 整数位置，可有非零偏移。采用相邻偶奇维配对（interleaved），不要采用 split-half。返回同形张量。',
    'x, positions, base=10000.0','θᵢ = base^(−2i/D)\n(x₂ᵢ,x₂ᵢ₊₁) 旋转 position · θᵢ',
    ['angles 的形状是 [T,D/2]。','旋转后按偶奇交错恢复原形。'],rope,
    lambda:[('连续位置',(rand(2,3,4,6),torch.arange(4))),('解码偏移',(rand(1,2,2,8),torch.tensor([10,11]))),('零位置',(rand(2,1,1,4),torch.tensor([0])))],source='https://arxiv.org/abs/2104.09864')

def swiglu(x,wgate,wup,wdown):
    gate=x@wgate
    return (gate*torch.sigmoid(gate)*(x@wup))@wdown
add('swiglu','SwiGLU 前馈网络','基础算子','基础',20,'x: [B,T,D]；wgate,wup: [D,F]；wdown: [F,D]。不含 bias、dropout。返回 [B,T,D]。禁止 silu。','x, wgate, wup, wdown','y = (SiLU(xWgate) ⊙ xWup)Wdown',['SiLU(z) = z · sigmoid(z)。'],swiglu,lambda:[('扩展宽度',(rand(2,3,4),rand(4,7),rand(4,7),rand(7,4))),('单 token',(rand(1,1,3),rand(3,5),rand(3,5),rand(5,3)))],source='https://arxiv.org/abs/2002.05202',forbidden=('silu',))

def ppo(new_logp,old_logp,advantages,clip_eps=0.2):
    ratio=(new_logp-old_logp.detach()).exp()
    a=advantages.detach()
    return -torch.minimum(ratio*a,ratio.clamp(1-clip_eps,1+clip_eps)*a).mean()
add('ppo','PPO：裁剪策略损失','强化学习','进阶',30,
    'new_logp、old_logp、advantages: [N]。计算最小化形式的 clipped policy loss；old_logp 与 advantages 必须 detach。仅实现策略项，不含 critic/entropy。clip_eps 在 (0,1)。',
    'new_logp, old_logp, advantages, clip_eps=0.2','L = −mean(min(rA, clip(r,1−ε,1+ε)A))\nr = exp(log πnew − log πold)',
    ['优势为负时仍使用 minimum，不能直接裁剪 ratio 后求均值。','旧策略与优势视为固定数据。'],ppo,
    lambda:[('正负优势与裁剪边界',(torch.tensor([0.,0.5,-0.7,0.8,-0.8],dtype=torch.float64),torch.zeros(5,dtype=torch.float64),torch.tensor([1.,2.,3.,-2.,-3.],dtype=torch.float64))),('自定义 epsilon',(rand(8)*0.2,rand(8)*0.2,rand(8),0.1))],source='https://arxiv.org/abs/1707.06347')

def gae(rewards,values,dones,gamma=0.99,lam=0.95):
    with torch.no_grad():
        adv=torch.zeros_like(rewards)
        carry=torch.zeros_like(rewards[0])
        for t in range(rewards.shape[0]-1,-1,-1):
            alive=1-dones[t].to(rewards.dtype)
            delta=rewards[t]+gamma*values[t+1]*alive-values[t]
            carry=delta+gamma*lam*alive*carry
            adv[t]=carry
        returns=adv+values[:-1]
    return adv,returns
add('gae','PPO：广义优势估计 GAE','强化学习','进阶',35,
    'rewards、dones: [T,B]；values: [T+1,B]，最后一行用于 bootstrap。dones[t] 表示这一步后环境真正终止（不是时间截断）。反向递推，终止处切断。返回 (advantages, returns)，均 [T,B] 且不带梯度。',
    'rewards, values, dones, gamma=0.99, lam=0.95','δt = rt + γ(1−doneₜ)Vt₊₁ − Vt\nAₜ = δt + γλ(1−doneₜ)Aₜ₊₁',
    ['从最后一步反向计算。','return target = advantage + value。'],gae,
    lambda:[('中途终止',(rand(4,2),rand(5,2),torch.tensor([[0,0],[1,0],[0,1],[0,0]],dtype=torch.bool))),('单步终止',(rand(1,2),rand(2,2),torch.ones(1,2,dtype=torch.bool))),('lambda=0',(rand(3,1),rand(4,1),torch.zeros(3,1,dtype=torch.bool),0.9,0.0))],source='https://arxiv.org/abs/1506.02438')

def grpo_adv(rewards,eps=1e-6):
    r=rewards.detach()
    return (r-r.mean(dim=-1,keepdim=True))/(r.std(dim=-1,keepdim=True,unbiased=False)+eps)
add('grpo-advantages','GRPO：组内优势归一化','强化学习','基础',20,
    'rewards: [B,G]，每行是同一 prompt 的 G 个回答奖励。本题约定总体标准差 unbiased=False，分母为 std + eps。返回同形优势并停止梯度。G=1 或组内奖励相同应返回 0。',
    'rewards, eps=1e-6','Aᵢ = (rᵢ − mean_group(r)) / (std_group(r) + ε)',
    ['沿 group 维归一化，不能把不同 prompt 混在一起。','这里使用总体标准差，是题库明确选择的约定。'],grpo_adv,
    lambda:[('多个 prompt',(rand(3,5),)),('同奖励',(torch.ones(2,4,dtype=torch.float64),)),('G=1',(rand(3,1),))],source='https://arxiv.org/abs/2402.03300')

def grpo(new_logp,old_logp,ref_logp,advantages,mask,clip_eps=0.2,beta=0.04):
    a=advantages.detach().unsqueeze(-1)
    ratio=(new_logp-old_logp.detach()).exp()
    obj=torch.minimum(ratio*a,ratio.clamp(1-clip_eps,1+clip_eps)*a)
    delta=ref_logp.detach()-new_logp
    kl=delta.exp()-delta-1
    loss=-(obj-beta*kl)*mask
    lengths=mask.sum(dim=-1)
    valid=lengths>0
    return (loss.sum(dim=-1)/lengths.clamp_min(1)*valid).sum()/valid.sum().clamp_min(1)
add('grpo','GRPO：带 KL 的 token 策略损失','强化学习','挑战',45,
    'new_logp、old_logp、ref_logp、mask: [B,G,T]；advantages: [B,G] 为已算好的组内优势。mask 为 bool，True 表示有效 token。先对每条回答有效 token 求均值，再对非空回答求均值。全空返回可求导 0。old/ref/advantages 停止梯度。使用下述 k3 KL 估计。本题不实现采样、奖励模型与优化器。',
    'new_logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.04','k3 = exp(log πref − log πnew) − (log πref − log πnew) − 1\nL = −mean_response mean_token(min(rA, clip(r)A) − β · k3)',
    ['不要对所有 token 直接求一次平均，否则长回答权重更大。','组优势在 token 维广播；空回答不参与外层平均。'],grpo,
    lambda:[('不同回答长度',(rand(1,3,4)*.2,rand(1,3,4)*.2,rand(1,3,4)*.2,rand(1,3),torch.tensor([[[1,1,1,1],[1,0,0,0],[0,0,0,0]]],dtype=torch.bool))),('多个组',(rand(2,3,2)*.2,rand(2,3,2)*.2,rand(2,3,2)*.2,rand(2,3),torch.ones(2,3,2,dtype=torch.bool))),('全空',(rand(1,2,3)*.2,rand(1,2,3)*.2,rand(1,2,3)*.2,rand(1,2),torch.zeros(1,2,3,dtype=torch.bool)))],source='https://arxiv.org/abs/2402.03300')

def dpo(chosen,rejected,ref_chosen,ref_rejected,beta=0.1):
    z=beta*((chosen-rejected)-(ref_chosen.detach()-ref_rejected.detach()))
    return torch.logaddexp(torch.zeros_like(z),-z).mean()
add('dpo','DPO：偏好优化损失','强化学习','进阶',25,
    '四个输入均为 [B] 的序列 log probability 之和。chosen/rejected 属于当前策略，ref_* 属于参考策略且须停止梯度。返回 batch 平均损失，需支持较大的 log probability 差。',
    'chosen, rejected, ref_chosen, ref_rejected, beta=0.1','L = −mean log σ(β[(log πw − log πl) − (log πref,w − log πref,l)])',
    ['用 logaddexp 或稳定 log-sigmoid。','两个当前策略输入都应保留梯度。'],dpo,
    lambda:[('偏好对',(rand(5),rand(5),rand(5),rand(5))),('大幅差值',(rand(3)*1000,rand(3)*1000,rand(3)*1000,rand(3)*1000,0.3))],source='https://arxiv.org/abs/2305.18290')


from extra_catalog import register
register(add, rand, PROBLEMS)
from templates import add_templates
add_templates(PROBLEMS)

def preview(x):
    if isinstance(x, torch.Tensor):
        values=x.detach().flatten()[:8].tolist()
        values=[round(v,5) if isinstance(v,float) else v for v in values]
        return str(values) + (' …' if x.numel()>8 else '')
    if isinstance(x,(tuple,list)): return ' | '.join(preview(v) for v in x)
    return str(x)


def public_problem(p,detail=False):
    keys=['id','title','category','difficulty','minutes']
    if detail: keys+=['description','signature','formula','hints','source','starter','runner','inference_only']
    result={k:p[k] for k in keys}
    if detail:
        torch.manual_seed(17)
        cases=p['cases']()
        result['examples']=[]
        for name,args in cases[:2]:
            with torch.no_grad(): output=p['reference'](*args)
            result['examples'].append({'name':name,'inputs':[{'shape':list(x.shape),'dtype':str(x.dtype),'preview':preview(x)} if isinstance(x,torch.Tensor) else x for x in args], 'output':preview(output)})
        result['testCount']=len(cases)
    return result
