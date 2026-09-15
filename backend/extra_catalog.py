"""Metrics, normalization and incremental decoding exercises."""
import torch


def register(add, rand, problems):
    metrics_source='https://scikit-learn.org/stable/modules/model_evaluation.html'
    def metric_cases():
        return [('普通二分类',(torch.tensor([0.9,0.6,0.7,0.2,0.5],dtype=torch.float64),torch.tensor([1,0,1,1,0]))),
                ('无预测正类',(torch.tensor([0.1,0.2,0.3],dtype=torch.float64),torch.tensor([1,0,1]))),
                ('无真实正类',(torch.tensor([0.9,0.1,0.5],dtype=torch.float64),torch.zeros(3,dtype=torch.long))),
                ('自定义阈值',(torch.rand(12,dtype=torch.float64),torch.randint(0,2,(12,)),0.7))]
    def accuracy(scores,targets,threshold=0.5):
        return ((scores>=threshold)==targets.bool()).double().mean()
    def precision(scores,targets,threshold=0.5):
        pred=scores>=threshold
        tp=(pred & targets.bool()).sum().double()
        return tp/pred.sum().clamp_min(1)
    def recall(scores,targets,threshold=0.5):
        pred=scores>=threshold
        tp=(pred & targets.bool()).sum().double()
        return tp/targets.sum().clamp_min(1)
    def f1(scores,targets,threshold=0.5):
        pred=scores>=threshold
        tp=(pred & targets.bool()).sum().double()
        return 2*tp/(pred.sum()+targets.sum()).clamp_min(1)
    for id,title,fn,formula in [
        ('accuracy','Accuracy：二分类准确率',accuracy,'Accuracy = (TP + TN) / N'),
        ('precision','Precision：精确率',precision,'Precision = TP / (TP + FP)'),
        ('recall','Recall：召回率',recall,'Recall = TP / (TP + FN)'),
        ('f1','F1：二分类调和平均',f1,'F1 = 2TP / (2TP + FP + FN)')]:
        add(id,title,'评价指标','入门',15,
            'scores: [N] 概率，targets: [N] 的 long 类型 0/1 标签，N>0。scores >= threshold 预测为正类（含等号）。返回标量 Tensor；分母为 0 时返回 0。本类题目仅检查数值，无需梯度。',
            'scores, targets, threshold=0.5',formula,['先计算预测布尔掩码。','统计 TP、FP、FN，明确零分母约定。'],fn,metric_cases,source=metrics_source)
    def confusion(pred,targets,num_classes):
        return torch.bincount(targets*num_classes+pred,minlength=num_classes**2).reshape(num_classes,num_classes)
    def class_cases():
        return [('含缺失类别',(torch.tensor([0,1,1,2,0]),torch.tensor([0,2,1,2,1]),4)),
                ('随机多分类',(torch.randint(0,5,(17,)),torch.randint(0,5,(17,)),5)),
                ('单一类别',(torch.tensor([1,1,1]),torch.tensor([1,1,1]),3))]
    add('confusion-matrix','多分类混淆矩阵','评价指标','基础',20,
        'pred、targets: [N] long 标签，值域 [0,num_classes)。返回 [C,C] 计数 Tensor，行是真实类别，列是预测类别。必须保留本批次缺失的类别。',
        'pred, targets, num_classes','M[i,j] = count(target=i, prediction=j)',
        ['将二维类别坐标编码成 target*C+pred。','bincount 需要 minlength=C*C。'],confusion,class_cases,source=metrics_source)
    def macro_f1(pred,targets,num_classes):
        labels=torch.arange(num_classes,device=pred.device)
        predicted=pred[:,None]==labels
        actual=targets[:,None]==labels
        tp=(predicted & actual).sum(0).double()
        f1=2*tp/(predicted.sum(0)+actual.sum(0)).clamp_min(1)
        return f1.mean()
    add('macro-f1','Macro-F1：多分类宏平均','评价指标','基础',25,
        'pred、targets: [N] long 标签，num_classes=C。分别计算全部 C 个类别的 F1，再等权平均。某类分母为 0 时该类 F1=0，仍参与平均。返回标量 Tensor；不是先平均 Precision/Recall 再计算 F1。',
        'pred, targets, num_classes','Macro-F1 = (1/C) Σc 2TPc / (2TPc + FPc + FNc)',
        ['类别缺失时也要保留该类。','macro 与 micro 在不平衡数据上不同。'],macro_f1,class_cases,source=metrics_source)
    def topk(logits,targets,k=1):
        indices=logits.topk(k,dim=-1).indices
        return (indices==targets[:,None]).any(dim=-1).double().mean()
    add('topk-accuracy','Top-k Accuracy','评价指标','基础',15,
        'logits: [N,C]，targets: [N] long，1<=k<=C，各行 logits 不含并列值。真实类别在前 k 个预测中即为命中，返回命中率标量 Tensor。不要先 softmax。',
        'logits, targets, k=1','Top-k Accuracy = mean(target ∈ topk(logits))',
        ['topk 沿类别维执行。','每条样本只计 0/1 次命中。'],topk,
        lambda:[('Top-1',(rand(5,4),torch.tensor([0,1,2,3,0]))),('Top-3',(rand(6,5),torch.tensor([0,1,2,3,4,0]),3)),('k=C',(rand(3,4),torch.tensor([1,2,3]),4))],source=metrics_source)
    def auc(scores,targets):
        pos=scores[targets==1]
        neg=scores[targets==0]
        diff=pos[:,None]-neg[None,:]
        return ((diff>0).double()+0.5*(diff==0).double()).mean()
    def ranking_cases():
        return [('混合排序',(torch.tensor([0.1,0.4,0.35,0.8],dtype=torch.float64),torch.tensor([0,0,1,1]))),
                ('并列分数',(torch.tensor([0.5,0.5,0.2,0.2,0.9],dtype=torch.float64),torch.tensor([1,0,1,0,1]))),
                ('全并列',(torch.ones(6,dtype=torch.float64),torch.tensor([1,1,0,1,0,0]))),
                ('反向排序',(torch.tensor([0.1,0.2,0.8,0.9],dtype=torch.float64),torch.tensor([1,1,0,0]))),
                ('随机排序',(rand(10),torch.tensor([0,1,0,1,0,1,0,1,0,1])))]
    add('roc-auc','ROC-AUC：排序与并列分数','评价指标','进阶',30,
        'scores: [N] 任意实数预测分数，targets: [N] long 0/1。保证同时存在正负样本。返回 ROC 曲线面积标量 Tensor：正样本得分高于负样本记 1，相等记 0.5。不能二值化分数。允许 O(N²) 成对比较，进阶可实现平均秩 O(N log N)。',
        'scores, targets','AUC = P(s_positive > s_negative) + 0.5 P(s_positive = s_negative)',
        ['成对比较的矩阵为 [N_positive,N_negative]。','并列分数不能依赖排序的先后顺序。'],auc,ranking_cases,
        source='https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html')
    def ap(scores,targets):
        thresholds=torch.unique(scores).sort(descending=True).values
        positives=targets.sum().double()
        result=torch.zeros((),dtype=torch.float64,device=scores.device)
        previous=torch.zeros_like(result)
        for threshold in thresholds:
            predicted=scores>=threshold
            tp=(predicted & targets.bool()).sum().double()
            recall=tp/positives.clamp_min(1)
            precision=tp/predicted.sum()
            result=result+(recall-previous)*precision
            previous=recall
        return result
    add('average-precision','Average Precision：PR 曲线','评价指标','进阶',30,
        'scores: [N] 分数，targets: [N] long 0/1。按不同分数阈值降序，将并列分数一起纳入，计算 AP=Σ(Recall增量×当前Precision)。无正样本返回 0。这是非插值 AP，不是对 PR 曲线做梯形积分。',
        'scores, targets','AP = Σn (Rn − Rn−1) Pn',
        ['每个唯一分数对应一个阈值，不能逐个拆开并列项。','初始 recall 为 0。'],ap,
        lambda:ranking_cases()+[('无正类',(rand(4),torch.zeros(4,dtype=torch.long)))],
        source='https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html')
    def iou(pred,target):
        intersection=(pred & target).sum().double()
        union=(pred | target).sum()
        return torch.where(union>0,intersection/union.clamp_min(1),torch.ones_like(intersection))
    def dice(pred,target):
        intersection=(pred & target).sum().double()
        total=pred.sum()+target.sum()
        return torch.where(total>0,2*intersection/total.clamp_min(1),torch.ones_like(intersection))
    def mask_cases():
        return [('部分交集',(torch.tensor([[1,1],[0,0]],dtype=torch.bool),torch.tensor([[1,0],[1,0]],dtype=torch.bool))),
                ('双空掩码',(torch.zeros(2,3,dtype=torch.bool),torch.zeros(2,3,dtype=torch.bool))),
                ('随机三维',(torch.rand(2,3,4)>.5,torch.rand(2,3,4)>.5))]
    for id,title,fn,formula in [('iou','IoU：分割交并比',iou,'IoU = |P ∩ Y| / |P ∪ Y|'),('dice','Dice：分割相似系数',dice,'Dice = 2|P ∩ Y| / (|P| + |Y|)')]:
        add(id,title,'评价指标','基础',15,'pred、target 为同形 bool 张量。所有维度合并统计一个整体指标，返回标量 Tensor。本题约定双空掩码得分为 1；仅一侧为空得分为 0。',
            'pred, target',formula,['使用布尔集合运算。','双空掩码是显式约定，不要直接除以 0。'],fn,mask_cases,source=metrics_source)

    # Preserve existing IDs/drafts while giving normalization its own category.
    for id in ('layernorm','rmsnorm'): problems[id]['category']='Normalization'
    def batchnorm(x,weight,bias,running_mean,running_var,training=True,momentum=0.1,eps=1e-5):
        axes=(0,)+tuple(range(2,x.ndim))
        shape=(1,x.shape[1])+((1,)*(x.ndim-2))
        if training:
            mean=x.mean(dim=axes)
            var=x.var(dim=axes,unbiased=False)
            n=x.numel()//x.shape[1]
            new_mean=(1-momentum)*running_mean.detach()+momentum*mean.detach()
            new_var=(1-momentum)*running_var.detach()+momentum*var.detach()*n/(n-1)
        else:
            mean,var=running_mean.detach(),running_var.detach()
            new_mean,new_var=mean.clone(),var.clone()
        y=(x-mean.reshape(shape))*torch.rsqrt(var.reshape(shape)+eps)
        return y*weight.reshape(shape)+bias.reshape(shape),new_mean,new_var
    def bn_cases(two_d):
        shape=(3,2,2,3) if two_d else (4,2)
        args=(rand(*shape),rand(2),rand(2),rand(2),torch.rand(2,dtype=torch.float64)+.5)
        const=(torch.ones(shape,dtype=torch.float64),)+args[1:]
        more_shape=(1,2,2,2) if two_d else (2,2,3)
        return [('训练统计量',args),('推理读取 running stats',args+(False,)),('常量通道',const),('不同维度 / momentum=1',(rand(*more_shape),)+args[1:]+(True,1.0))]
    for id,title,two in [('batchnorm1d','BatchNorm1d：训练与推理',False),('batchnorm2d','BatchNorm2d：通道统计与滑动均值',True)]:
        add(id,title,'Normalization','进阶',35,
            ('x: [N,C] 或 [N,C,L]。' if not two else 'x: [N,C,H,W]。')+
            'weight、bias、running_mean、running_var 均 [C]。训练时沿非通道轴统计：前向使用总体方差 unbiased=False，更新 running_var 使用无偏方差；每通道元素数>1。更新为 (1-momentum)*旧值+momentum*新值。推理只用传入的 running stats。返回 (y,new_running_mean,new_running_var)，后两个停止梯度，禁止原地修改输入。禁止调用现成 BatchNorm。',
            'x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-5',
            'y = (x − μ) / √(var_biased + ε) · γ + β\nrunning_var ← (1−m)running_var + m · var_unbiased',
            ['训练归一化方差与 running_var 更新方差不同。','参数广播形状为 [1,C,1,…]。','推理时不要再次计算 batch 均值。'],batchnorm,lambda t=two:bn_cases(t),
            source='https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html',forbidden=('batch_norm','BatchNorm1d','BatchNorm2d'))
    def instance(x,weight,bias,eps=1e-5):
        mean=x.mean(dim=(2,3),keepdim=True)
        var=x.var(dim=(2,3),keepdim=True,unbiased=False)
        return (x-mean)*torch.rsqrt(var+eps)*weight[None,:,None,None]+bias[None,:,None,None]
    add('instancenorm','InstanceNorm2d：逐样本逐通道','Normalization','基础',25,
        'x: [N,C,H,W]，weight、bias: [C]。各样本各通道独立沿 H,W 归一化，使用总体方差；无 running stats，训练与推理相同。返回同形 Tensor，禁止 instance_norm/InstanceNorm2d。',
        'x, weight, bias, eps=1e-5','μnc = mean_HW(xnc), σ²nc = var_HW(xnc)',
        ['不能跨 batch 或 channel 统计。','eps 在根号内。'],instance,
        lambda:[('不同样本分布',(rand(3,2,2,4),rand(2),rand(2))),('常量通道',(torch.ones(2,3,2,2,dtype=torch.float64),rand(3),rand(3))),('单样本',(rand(1,4,3,2),rand(4),rand(4)))],
        source='https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html',forbidden=('instance_norm','InstanceNorm2d'))
    def group(x,weight,bias,num_groups,eps=1e-5):
        n,c,h,w=x.shape
        grouped=x.reshape(n,num_groups,-1)
        centered=grouped-grouped.mean(dim=-1,keepdim=True)
        normalized=centered*torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+eps)
        return normalized.reshape_as(x)*weight[None,:,None,None]+bias[None,:,None,None]
    add('groupnorm','GroupNorm：分组归一化','Normalization','进阶',25,
        'x: [N,C,H,W]；weight、bias: [C]；C 能被 num_groups 整除。每个样本的连续 C/G 个通道构成一组，沿组内通道与空间归一化。总体方差，无 running stats，返回同形 Tensor。禁止 group_norm/GroupNorm。',
        'x, weight, bias, num_groups, eps=1e-5','reshape: [N,C,H,W] → [N,G,(C/G)HW]',
        ['G=1 跨全部通道和空间归一化。','G=C 时等价于本题 InstanceNorm。'],group,
        lambda:[('G=2',(rand(2,6,2,3),rand(6),rand(6),2)),('G=1',(rand(2,4,2,2),rand(4),rand(4),1)),('G=C',(rand(2,3,2,4),rand(3),rand(3),3))],
        source='https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html',forbidden=('group_norm','GroupNorm'))
    def layer_nd(x,weight,bias,normalized_shape,eps=1e-5):
        axes=tuple(range(x.ndim-len(normalized_shape),x.ndim))
        mean=x.mean(dim=axes,keepdim=True)
        variance=x.var(dim=axes,keepdim=True,unbiased=False)
        return (x-mean)*torch.rsqrt(variance+eps)*weight+bias
    add('layernorm-nd','LayerNorm：多维 normalized_shape','Normalization','进阶',25,
        'normalized_shape 是与 x 尾部若干维度完全匹配的 tuple；weight、bias 形状等于 normalized_shape。沿这些尾部维度联合归一化，使用总体方差。返回与 x 同形 Tensor，禁止 layer_norm/LayerNorm。',
        'x, weight, bias, normalized_shape, eps=1e-5','归一化轴 = 最后 len(normalized_shape) 个维度',
        ['(C,H,W) 的 LayerNorm 不等于逐通道 BatchNorm。','仿射参数也覆盖全部 normalized_shape。'],layer_nd,
        lambda:[('最后一维',(rand(2,3,4),rand(4),rand(4),(4,))),('最后两维',(rand(2,3,4),rand(3,4),rand(3,4),(3,4))),('图像三维',(rand(2,3,2,2),rand(3,2,2),rand(3,2,2),(3,2,2)))],
        source='https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html',forbidden=('layer_norm','LayerNorm'))
    def l2(x,dim=-1,eps=1e-12):
        return x/torch.linalg.vector_norm(x,ord=2,dim=dim,keepdim=True).clamp_min(eps)
    add('l2-normalize','L2 向量归一化','Normalization','基础',15,
        'x 为浮点 Tensor。沿 dim 计算 L2 范数，分母为 max(norm,eps)，不是 sqrt(sum(x²)+eps)。返回同形 Tensor，全零向量输出 0。禁止 normalize。',
        'x, dim=-1, eps=1e-12','y = x / max(‖x‖₂, ε)',
        ['保留归约轴便于广播。','测试会区分 clamp_min(eps) 与根号内加 eps。'],l2,
        lambda:[('最后一维',(rand(3,4),)),('零向量',(torch.zeros(2,4,dtype=torch.float64),)),('指定轴与 epsilon',(rand(2,3,4)*.01,1,.1))],
        source='https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html',forbidden=('normalize',))

    cache_source='https://huggingface.co/docs/transformers/cache_explanation'
    def append_cache(past_k,past_v,new_k,new_v):
        return torch.cat((past_k,new_k),dim=2),torch.cat((past_v,new_v),dim=2)
    def append_cases():
        return [('空缓存 prefill',(rand(2,3,0,4),rand(2,3,0,5),rand(2,3,3,4),rand(2,3,3,5))),
                ('单 token decode',(rand(1,2,5,3),rand(1,2,5,4),rand(1,2,1,3),rand(1,2,1,4))),
                ('多 token 追加',(rand(2,2,3,4),rand(2,2,3,2),rand(2,2,2,4),rand(2,2,2,2)))]
    add('kv-cache-append','KV Cache：追加与 Prefill','KV Cache','基础',20,
        'past_k: [B,H,P,Dk]，past_v: [B,H,P,Dv]；new_k: [B,H,T,Dk]，new_v: [B,H,T,Dv]。返回 (all_k,all_v)，按序列维先旧后新拼接。支持 P=0，禁止原地修改输入。缓存系列按推理模式判题，不检查梯度。',
        'past_k, past_v, new_k, new_v','cache_length_new = P + T',
        ['缓存序列轴是 dim=2，不是最后一轴。','K 与 V 的最后一维可以不同。'],append_cache,append_cases,source=cache_source)
    def decode(q,new_k,new_v,past_k,past_v):
        all_k=torch.cat((past_k,new_k),dim=2)
        all_v=torch.cat((past_v,new_v),dim=2)
        past=past_k.shape[2]
        t=q.shape[2]
        visible=torch.arange(past+t,device=q.device)[None,:] <= past+torch.arange(t,device=q.device)[:,None]
        s=q@all_k.transpose(-1,-2)/q.shape[-1]**0.5
        output=torch.softmax(s.masked_fill(~visible,float('-inf')),dim=-1)@all_v
        return output,all_k,all_v
    def decode_cases(grouped=False):
        rows=[]
        for label,p,t,h,hkv in [('空缓存',0,4,4,2),('单 token 解码',5,1,4,1),('分块解码偏移',3,3,6,3)]:
            hk=hkv if grouped else h
            rows.append((label,(rand(2,h,t,4),rand(2,hk,t,4),rand(2,hk,t,3),rand(2,hk,p,4),rand(2,hk,p,3))))
        return rows
    add('kv-cache-decode','KV Cache：增量因果注意力','KV Cache','进阶',35,
        'q,new_k: [B,H,T,Dk]，new_v: [B,H,T,Dv]，past_k: [B,H,P,Dk]，past_v: [B,H,P,Dv]。先追加缓存；新查询 i 的绝对位置为 P+i，只能看 key 0…P+i。返回 (output,all_k,all_v)，output: [B,H,T,Dv]。支持空缓存、单 token 与分块解码。输入 Q/K 已完成位置编码，不得再次旋转。禁止修改输入。',
        'q, new_k, new_v, past_k, past_v','visible[i,j] = (j ≤ P + i)',
        ['直接对 [T,P+T] 使用 tril() 会漏掉历史 token。','追加后的缓存必须同时返回供下一步使用。'],decode,decode_cases,source=cache_source)
    def gqa_decode(q,new_k,new_v,past_k,past_v):
        all_k=torch.cat((past_k,new_k),dim=2)
        all_v=torch.cat((past_v,new_v),dim=2)
        repeats=q.shape[1]//all_k.shape[1]
        k=all_k.repeat_interleave(repeats,dim=1)
        v=all_v.repeat_interleave(repeats,dim=1)
        p,t=past_k.shape[2],q.shape[2]
        visible=torch.arange(p+t,device=q.device)[None,:]<=p+torch.arange(t,device=q.device)[:,None]
        scores=q@k.transpose(-1,-2)/q.shape[-1]**0.5
        return torch.softmax(scores.masked_fill(~visible,float('-inf')),dim=-1)@v,all_k,all_v
    add('kv-cache-gqa','KV Cache：GQA / MQA 共享缓存','KV Cache','挑战',40,
        'q: [B,Hq,T,Dk]；new_k/v 与 past_k/v 分别 [B,Hkv,T/P,Dk或Dv]。Hq 能被 Hkv 整除，连续 query heads 共享一个 KV head。执行带位置偏移的增量因果注意力。返回 (output,all_k,all_v)；输出 Hq 个头，但持久缓存始终只有 Hkv 个头，不能存重复后的 K/V。Q/K 已编码，输入不可修改。',
        'q, new_k, new_v, past_k, past_v','计算时广播 KV；缓存保留 Hkv，而不是 Hq',
        ['扩展仅用于 attention 计算。','Hkv=1 就是 MQA 缓存。'],gqa_decode,lambda:decode_cases(True),source=cache_source)
    def sliding(q,new_k,new_v,past_k,past_v,window_size):
        k=torch.cat((past_k,new_k),dim=2)
        v=torch.cat((past_v,new_v),dim=2)
        p,t=past_k.shape[2],q.shape[2]
        positions=p+torch.arange(t,device=q.device)[:,None]
        keys=torch.arange(p+t,device=q.device)[None,:]
        visible=(keys<=positions)&(keys>positions-window_size)
        scores=q@k.transpose(-1,-2)/q.shape[-1]**0.5
        y=torch.softmax(scores.masked_fill(~visible,float('-inf')),dim=-1)@v
        return y,k[:,:,-window_size:,:],v[:,:,-window_size:,:]
    add('kv-cache-sliding','KV Cache：滑动窗口与分块解码','KV Cache','挑战',40,
        'q/new_k: [B,H,T,Dk]，new_v: [B,H,T,Dv]；past_k/v: [B,H,P,Dk或Dv]，历史缓存已连续截断且 P<=window_size。每个查询只看包括自身在内最近 window_size 个 token。返回 (output,next_k,next_v)，新缓存保留最后 min(P+T,window_size) 个 token。Q/K 已编码。先为整个新块计算各自窗口，再裁剪返回缓存；不可先统一截断导致块前部查询丢失上下文。输入不可修改。',
        'q, new_k, new_v, past_k, past_v, window_size','P+i−window_size < key_index ≤ P+i',
        ['块内不同查询的左边界不同。','window_size=1 时只能关注自身。'],sliding,
        lambda:[('空缓存',(rand(1,2,3,4),rand(1,2,3,4),rand(1,2,3,2),rand(1,2,0,4),rand(1,2,0,2),2)),
                ('分块窗口移动',(rand(2,2,3,4),rand(2,2,3,4),rand(2,2,3,3),rand(2,2,4,4),rand(2,2,4,3),4)),
                ('窗口为 1',(rand(1,2,3,4),rand(1,2,3,4),rand(1,2,3,2),rand(1,2,1,4),rand(1,2,1,2),1))],source=cache_source)
    for p in problems.values():
        p['inference_only']=p['category'] in ('评价指标','KV Cache')
        p['immutable_inputs']=p['category'] in ('Normalization','KV Cache')
