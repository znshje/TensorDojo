# 题库维护、导入与导出

题库由内置题目和本地持久化覆盖层组成。**日常维护使用 API 或命令行，不必编辑 `catalog.py`。** 维护成功后服务立即生效，页面刷新即可看到；已有草稿、提交历史不会被删除。

## 快速开始

先运行 `./start.sh`。以下命令均在项目根目录执行，命令行客户端只需要 Python 标准库，无需安装 PyTorch：

```bash
# 查看版本、有效题目、已隐藏题目
python scripts/bank_cli.py list

# 查看单题完整定义（包括参考实现和测试数据）
python scripts/bank_cli.py get f1

# 全量备份 / 按 ID 导出
python scripts/bank_cli.py export -o question-bank-backup.json
python scripts/bank_cli.py export --ids f1,roc-auc -o metrics.json

# 获取一份可以修改的完整示例
python scripts/bank_cli.py example -o my-question.json

# 只检查格式、参考答案和测试，不写入题库
python scripts/bank_cli.py validate my-question.json

# 预演导入，同时检查同名题目冲突
python scripts/bank_cli.py import my-question.json --dry-run

# 正式导入：默认拒绝覆盖同名题目
python scripts/bank_cli.py import my-question.json

# 编辑 JSON 后更新（文件中只能有一道题）
python scripts/bank_cli.py update my-question.json

# 多题更新需显式允许覆盖；不会移除文件中未包含的题目
python scripts/bank_cli.py import metrics.json --overwrite

# 隐藏题目 / 恢复原定义
python scripts/bank_cli.py delete demo-l1-normalize
python scripts/bank_cli.py restore demo-l1-normalize
```

`create FILE` 新建单题；`update FILE` 全量替换单题，不是字段合并。两个命令都可接受单题对象、`get` 的返回对象，或仅含一道题的 bundle。

`export`、`example` 默认不覆盖已有文件，覆盖时显式使用 `--force`。省略 `-o` 输出 JSON 到标准输出。自定义服务地址放在子命令前：

```bash
python scripts/bank_cli.py --url http://127.0.0.1:8766 list
```

新增的题目可以使用任意中文分类名，页面会自动显示新分类。为保持草稿和历史关联，更新时保持 `id` 不变；更换 ID 相当于新增另一道题。

## HTTP API

基础地址：`http://127.0.0.1:8765`。

先 `GET /api/health` 取得当前会话 `token`。所有题库管理请求须提供 `X-Dojo-Token: <token>`；写请求需 `Content-Type: application/json`。服务重启后 token 会变化，CLI 每次自动获取。

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/api/bank` | 返回全局 `revision`、有效题目摘要、已删除 ID |
| GET | `/api/bank/problems/{id}` | 单题完整定义及当前 `revision` |
| POST | `/api/bank/problems` | 新增单题 |
| PUT | `/api/bank/problems/{id}` | 完整替换现有题目，不能改 ID |
| DELETE | `/api/bank/problems/{id}` | 可恢复地隐藏题目 |
| POST | `/api/bank/problems/{id}/restore` | 恢复已隐藏题目 |
| GET | `/api/bank/export` | 导出全部有效题目 |
| GET | `/api/bank/export?ids=f1,roc-auc` | 导出指定题目 |
| GET | `/api/bank/example` | 获取完整可导入示例 |
| POST | `/api/bank/validate` | 校验 bundle，不写入，也不判断同名冲突 |
| POST | `/api/bank/import` | 校验并批量导入，支持预演与显式覆盖 |

### 新增 / 更新请求

```json
{
  "expected_revision": 0,
  "problem": {
    "id": "your-question",
    "title": "...完整题目定义，见 examples/question-bank.json..."
  }
}
```

上述 `problem` 仅展示外层结构，完整可运行内容见项目内 `examples/question-bank.json`。

### 导入请求

```json
{
  "expected_revision": 0,
  "overwrite": false,
  "dry_run": false,
  "bundle": {
    "format": "tensor-dojo-bank",
    "version": 1,
    "problems": []
  }
}
```

`problems` 须填入 1–200 道完整题目；不能提交空数组。`dry_run=true` 执行全部校验和冲突检查，但不会写入文件，也不提升 revision。

`validate` 请求只需 `{"bundle": <bundle对象>}`。

删除 / 恢复请求：`{"expected_revision": 3}`。

### 版本与原子性

每次成功写入，**全局** `revision` 加 1。所有写请求（包括导入预演）必须携带读取时的 `expected_revision`。服务会在校验前和写入前检查版本；不匹配时返回 `409`，重新获取版本并审核最新定义后重试。CLI 自动携带版本，但不会遇到冲突后盲目重试。

批量导入全成全败：任一题格式错误、参考答案不匹配、出现重复 ID 或覆盖冲突时，整个批次不写入。持久化先写临时文件并 fsync，然后原子替换题库文件。单个服务进程串行提交变更；不要启动多个进程同时写同一个题库文件。

响应示例：

```json
{
  "validated": 1,
  "ids": ["demo-l1-normalize"],
  "overwritten": [],
  "dry_run": false,
  "revision": 1
}
```

状态码：`200` 成功/预检；`201` 新增/无覆盖导入成功；`400` 格式或字段错误；`403` token/来源不合法；`404` 题目或接口不存在；`409` 同名冲突/版本冲突；`413` 请求超限；`422` 参考代码或测试校验失败/超时；`429` 已有导入校验在执行。

## JSON 交换格式 v1

参考完整文件：`examples/question-bank.json`。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 小写字母起始，字母/数字/连字符，最长 64 字符 |
| `title`, `category`, `description` | 是 | 标题、分类和清晰的输入输出约定 |
| `difficulty` | 是 | 入门 / 基础 / 进阶 / 挑战 |
| `minutes` | 是 | 1–600 整数 |
| `signature` | 是 | `solve` 参数列表，如 `x, dim=-1, eps=1e-12`；默认值只允许字面量，不支持可变参数/仅关键字参数 |
| `solution` | 是 | 完整 Python 参考代码，需定义顶层 `solve` |
| `tests` | 是 | 1–200 个具体测试，每项含 `name`、`args`、`expected` |
| `starter` | 否 | 练习代码模板，需定义顶层 `solve`；缺省生成 TODO 骨架 |
| `formula`, `hints`, `source` | 否 | 公式、字符串提示数组、http(s) 来源链接 |
| `forbidden` | 否 | 禁止直接调用的函数名数组，如 `["softmax"]` |
| `inference_only` | 否 | 默认 false；true 时只检查输出、不要求梯度 |
| `immutable_inputs` | 否 | 默认 false；true 时额外禁止原地修改输入 |
| `constant_indices` | 否 | 固定目标等不要求梯度的参数下标数组，默认 `[]` |

输出必须为 Tensor 或非空的 Tensor 元组。判题会比较张量数值、形状；需要梯度的题目还会对照参考实现的自动微分。`constant_indices` 是根本不启用梯度的参数；如果要测试旧策略等输入是否正确 `detach`，不要放进此数组，应在参考实现中明确停止梯度。

### 测试参数编码

JSON 标量直接使用 `true`、`false`、数字、字符串、`null`。Tensor 使用：

```json
{"type":"tensor","dtype":"float64","shape":[2,2],"data":[1,2,3,4]}
```

`data` 必须是扁平数组，数量严格等于 shape 的元素乘积。支持 `float32`、`float64`、`int32`、`int64`、`bool`。标量 Tensor 用 `shape: []` 和单元素 data；空张量可以用 `shape: [2,0,3], data: []`。不允许 NaN/Infinity，bool 张量必须使用 true/false。

Tuple 使用：

```json
{"type":"tuple","items":[3,4]}
```

多输出 expected 同样使用 tuple 编码，items 为 Tensor 编码。

### 测试、模板与可复现性

导出内置题目时，会展开原先的三个随机种子的测试并保存具体输入与期望值，因此导入到另一环境后无需原题的 Python 测试工厂。导入题目按文件中的测试各运行一次；页面显示实际测试数，不会把已展开的用例再乘三。前两个用例作为公开样例，第一个用于生成可独立运行的模板。

导入验证在独立子进程中执行参考实现，对照所有 `expected`，并检查参考实现能通过判题器及练习算子限制。缺少运行入口的 starter 会自动补齐输入、期望输出和 `__main__`。参考答案正确与否最终依赖维护者提供的预期结果，建议使用手算小样例和独立库实现交叉验证。

限制：单次请求 8 MiB，每题最多 200 个测试，每个张量最多 100000 元素；导入校验总时限 40 秒 / CPU 25 秒，超限时缩小批次。导入包含可执行 Python，因此只导入可信题库；现有执行器是个人本地工具，不是恶意代码安全沙箱。

## 存储、备份和源码扩展

- `data/question-bank.json`：自定义定义、内置覆盖、删除标记、缓存的公开题目与运行模板、全局 revision。
- `data/progress.sqlite3`：提交历史，独立于题库。删除题目不删除历史，恢复相同 ID 后再次关联。
- 完整恢复本机状态：停止服务后备份/恢复整个 `data/` 目录。`export` 只导出有效题目，不包含隐藏记录、草稿或成绩；重新导入是合并操作，不会删除目标库的其他题目。
- 若需独立测试环境，设置 `DOJO_DATA_DIR=/tmp/my-dojo-data` 后启动服务；也可用 `DOJO_BANK_PATH` 单独指定题库文件。
- 保留了原 `backend/catalog.py` / `extra_catalog.py` 的源码题库开发方式，适用于维护内置随机测试工厂。已有 API 覆盖优先于同 ID 内置定义。
- `scripts/export_templates.py` 会为当前有效题库导出缺失 `.py` 文件，已有本地练习文件不覆盖。网页下载始终使用最新题目版本。

验证命令：

```bash
/path/to/torch-python -m unittest discover -s tests -v
/path/to/torch-python scripts/check_bank_api.py
```

后一条会启动独立临时数据目录与临时端口，验证真实 HTTP CRUD、导入/导出、判题、重启持久化，不改动你的实际题库和成绩。

## 前端管理操作

打开页面右上角「题库管理」：

- **新建**：以可运行的 L1 示例起步，填写唯一 ID，修改题目、参考实现与测试，再点「校验并保存」。练习模板可留空自动生成。
- **编辑**：分别维护基本信息、参考实现、练习模板、测试数据和判题规则；ID 不可修改。
- **复制**：自动分配新 ID，进入编辑表单；保存后创建独立副本。
- **删除 / 恢复**：删除前确认，删除后可在「已删除题目」恢复；保留成绩和浏览器草稿。
- **导出**：每行导出单题，勾选后批量导出，或导出全部；JSON 包含参考实现与测试。
- **导入**：选择文件或粘贴 JSON，点击「预检导入」，核对新增与覆盖列表后确认。同 ID 默认报冲突，勾选允许覆盖后重新预检。

保存或导入成功后，练习列表即时更新。其他窗口修改题库导致版本冲突时，保留当前内容，返回列表刷新并重新打开题目后合并修改。
