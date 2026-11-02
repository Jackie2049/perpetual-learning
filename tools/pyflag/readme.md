# pyflag — 扫描"配置项有没有生产者"（布尔开关静默失效）

## 用在哪

一个布尔开关在源码里有**消费者**（函数形参 + `if flag:` 分支），却找不到任何
**生产者**（调用点传值），那它在本快照里不可达：用户设了也没用。这类问题不看
代码很难发现——开关名在 config、trainer、engine 里各出现一次，grep 起来"到处都在用"。

工具只做只读静态分析（AST），不导入、不运行目标仓库。

## 怎么用

```bash
python3 scan_unwired_flags.py /path/to/repo          # 打印 JSON 报告
python3 scan_unwired_flags.py /path/to/repo --json   # 同时落盘 scan-<repo>.json
python3 -m unittest test_scan_unwired_flags -v       # 10 个用例，含 4 条易错点
```

自动发现的开关：形参带 `= True/False` 默认值、且名字以 `use_` / `enable_` /
`disable_` / `no_` / `with_` 开头。

## 输出怎么读

每条 flag 给四个数和一个判定：

| 字段 | 含义 |
|---|---|
| `reads_nontest` | 非测试代码里出现该名字的**行数**（粗粒度，只作参照） |
| `consumers_nontest` | 把它当形参、且在函数体内**按它分支**的函数 |
| `producers_nontest` | 真的把值**传进去**的调用点（关键字或位置实参） |
| `silently_inert` | 有消费者、无生产者 |
| `mode` | `never` = 默认 False 且无人传 ⇒ **分支永不执行**；`always` = 默认 True 且无人传 ⇒ **分支总是执行**；`mixed` = 多个消费者默认值不一致 |

`mode` 这个区分很关键：同样是"没人传"，`never` 是功能静默失效，`always`
是功能静默生效（用户以为能关，其实关不掉），两者是完全不同的问题。

## 四个易错点（都真实踩过，测试里各有一条）

1. **构造器的调用点写的是类名**。消费者是 `Actor.__init__`，调用点是 `Actor(...)`。
   不处理会把所有类构造全部漏判。
2. **Ray 的 `.remote(...)` 包了一层**。`engine.init_group.remote(..., use_ray=x)`
   的真实被调方法是 `init_group`，不是 `remote`。
3. **同名局部变量覆盖形参**。函数体内 `use_ray = getattr(...)` 之后
   `if use_ray:` 读的是局部变量，不再构成"按该形参分支"。
4. **形参默认值不是生产者**。`def f(use_x=False)` 这一行里出现了名字，但没人调用它。

## 已知边界（结果必须带着这些读）

- **只有静态可达性判断**，不做数据流/跨进程分析。通过配置文件字典间接传参
  （`**cfg`、`getattr(ns, flag)`）**检测不到**，会漏报——本工具报"有生产者"时可信，
  报"无生产者"时仍需人工确认没有间接路径。
- **同名不同函数会串**。只按名字匹配调用点，不做符号解析。
- `reads_nontest` 是行数不是引用数，别当精度看。
- 测试/示例/文档目录按路径排除（`tests/`、`examples/`、`docs/` 等），
  但目录命名不统一时会漏。

## 已跑过的快照（2026-11-02）

| 仓库 | commit | 发现开关 | 判为失效 | 人工复核结论 |
|---|---|---|---|---|
| verl | `6879e1ab07` | 37 | 8 | `use_prefix_grouper`、`use_tiled_mlp` **确认为真**；`use_chunking_entropy` 也在同一批不可达路径上 |
| SkyRL | main | 39 | 7 | `use_liger_kernel`、`use_remove_padding` **确认为真** |
| OpenRLHF | main | 5 | 0 | 修掉三处假阳性后归零（详见测试用例） |

报告存档：`scan-verl-project-verl-6879e1ab07.json`、`scan-NovaSky-AI-SkyRL.json`、
`scan-OpenRLHF-OpenRLHF.json`。

## 判定的分量

"有消费者无生产者"是**必要条件**，不是充分条件。说"这个开关是死代码"之前，
至少要排除：① 间接传参（`**kwargs` / 配置对象属性）；② 插件或动态注册；
③ 下游 fork 的调用约定。本工具只负责把候选缩到几条，结论仍要人读代码。
