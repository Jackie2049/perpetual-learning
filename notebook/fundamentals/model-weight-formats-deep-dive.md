# 权重格式与映射加载

> 2026-06-12 | LLM权重存储格式全链路分析: safetensors(二进制规格+8B header+N JSON+data buffer+mmap/zero-copy/lazy load)+GGUF(llama.cpp+内置量化+CPU推理)+ONNX(跨框架+图+权重)+PyTorch pickle(训练checkpoint+安全风险)+vLLM weight loading pipeline(discover→detect→parse→map→allocate→transfer)+训练→分发→部署管线+RTX 4090
> 源码: github.com/huggingface/safetensors (Rust core + Python/Rust/Go bindings)
> 关联: checkpoint.md, vllm-weight-loading-reading.md, quantization.md, fp8-quantization.md

2026-10-26校正：**mmap返回Tensor不等于完整权重已从存储读入，更不等于模型已在GPU可用。** 撤回原文无来源的“0.1ms对30s、300000倍”和“8进程固定省98GB”结论；未将旧数值改称本机benchmark。下方GGUF／ONNX和模型管线仍是历史概念材料，未逐项重核其当前后端支持。

## 映射加载的实际边界

[PyTorch 2.14 torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)也支持 `mmap=True`，将storage字节的读取延后到访问时；并非只有safetensors能映射。安全策略、文件结构与加载策略是三个不同选择。`weights_only=True`缩小unpickler允许的对象集合，不代表checkpoint只有模型权重，也不等于可安全加载任意不可信文件，见[官方序列化说明](https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch-load-with-weights-only-true)。

固定[safetensors v0.8.0实现](https://github.com/huggingface/safetensors/blob/a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6/bindings/python/src/lib.rs)中，现代PyTorch CPU mmap路径采用 `UntypedStorage.from_file(..., shared=False)`；其他框架／pread／设备支路不能据此统称零拷贝。打开文件、解析header、构造Tensor、触及全部数据、dtype／布局转换、设备搬运、写入模型参数须分开计时；文件系统缓存及存储介质也要固定。[官方速度示例](https://huggingface.co/docs/safetensors/speed)有其旧软件／机器条件，不能外推成所有模型和路线的固定倍数。

[CPU文件控制](../../experiments/weight-file-mapping/verify.py)与[结果](../../experiments/weight-file-mapping/result.json)只用自建32KiB级Tensor文件：Python3.12.13／torch2.14.1／safetensors0.8.0，safetensors CPU mmap、torch.load CPU mmap和eager三路线均读到8192元素、求和33550336；将返回值首项改成−7后，重新打开仍为0，文件SHA256均未变。前两条与私有映射／写时复制语义相符，eager则本来有独立数据；**这项现象本身不能证明用了mmap或物理页共享**，机制还须结合源码。safetensors的Tensor在safe_open上下文关闭后仍可读，并非退出上下文就立即失效。

同一文件还将3个BOOL存为3数据字节、4个E4M3 FP8存为4字节，FP8原始字节回读一致；名为 `optimizer.momentum` 的浮点Tensor正常回读。格式本身不区分“模型权重”与“优化器Tensor”，但完整训练恢复仍需组织其余状态与结构。本控制没有计时、RSS／PSS、冷缓存、多进程、模型或设备测试；不以一次求和、映射字节数或文件大小替代这些证据。复现为 `python experiments/weight-file-mapping/verify.py --output result.json`，只创建并清理本次临时文件，无外部checkpoint或GPU访问。

## 0. 权重生命周期中的格式选择

```
LLM权重生命周期3阶段+对应格式:

阶段1: 训练 → PyTorch .pt/.bin (pickle)
  → 需要optimizer state+scheduler+RNG state等恢复信息 → .pt是常见方案，非唯一格式
  → 安全风险: pickle可执行任意代码 → 只在可信环境用!
  → 7B BF16约14GB参数；另存两份FP32 Adam矩约56GB是状态组成差异，不是格式固定大5倍

阶段2: 分发 → Safetensors (.safetensors)
  → 存命名Tensor与字符串metadata → 常用于分发模型；也能存优化器Tensor
  → mmap可延迟读取／减少某些复制 → 不承诺10GB模型毫秒级完整加载
  → 同文件私有映射可复用干净页；写入／转换／复制会改变共享情况
  → → HuggingFace Hub默认格式 → 2025标准!

阶段3: 部署 → GGUF/ONNX/TensorRT-LLM
  → GGUF → CPU/边缘推理 → llama.cpp → 内置量化 → 消费级硬件
  → ONNX → 跨框架+硬件优化 → ONNX Runtime → 多后端
  → TensorRT-LLM → NVIDIA GPU工具链之一 → 优化选择须匹配模型与实际工作负载
  → → 格式选择取决于目标硬件!

关键设计定律:
  → safetensors: Tensor结构明确，支持映射与选取；速度／内存仍依具体加载路线
  → GGUF: 模型元数据与量化表示 → 常见于llama.cpp生态
  → ONNX: 图+权重+跨框架 → 硬件加速 → 多后端推理
  → pickle: 可表达复杂对象，但非受限加载存在任意代码执行风险
  → → → 生命周期是使用场景，不是三种格式互不替代的硬规则
```

## 1. Safetensors 二进制格式规范

```
官方规范 (github.com/huggingface/safetensors README):

文件结构:
  [8字节] → N = header长度 (uint64 little-endian)
  [N字节] → JSON UTF-8 header → tensor metadata + __metadata__
  [剩余]  → byte-buffer → 连续tensor数据 → 无空洞!

Header JSON结构:
  {
    "__metadata__": {"format": "pt"},  // 自由字符串→字符串映射
    "model.layers.0.weight": {
      "dtype": "BF16",                  // 数据类型
      "shape": [4096, 4096],            // tensor形状
      "data_offsets": [0, 33554432]     // [START, END] → 字节偏移(相对byte-buffer开头)
    },
    "model.layers.1.weight": {
      "dtype": "BF16",
      "shape": [4096, 4096],
      "data_offsets": [33554432, 67108864]
    }
  }

dtype支持:
  → BOOL (每元素8bit) → U8 (uint8) → I8 (int8) → I16 (int16) → I32 (int32) → I64 (int64)
  → F16 (float16) → BF16 (bfloat16) → F32 (float32) → F64 (float64)
  → v0.8.0已定义F8_E4M3／F8_E5M2等；框架dtype与运行kernel支持另判
  → 本轮CPU只测E4M3原始字节回读，未验证任意FP8量化模型或推理后端

约束:
  → Header必须以{开头(0x7B) → 识别safetensors文件
  → Header可以尾部空格填充(0x20) → 8字节对齐!
  → data_offsets: [BEGIN, END] → END是one-past-end → tensor大小=END-BEGIN
  → byte-buffer必须完全索引 → 无空洞 → 防止polyglot文件(安全!)
  → 不允许重复key → 唯一tensor名
  → 允许空tensor(shape含0) → 不存数据 → header有size
  → 允许0-rank tensor(shape=[]) → scalar值
  → NaN/±Inf允许 → 不检查值 → 只检查结构!

7B模型safetensors文件示例:
  → Llama-2-7B: ~13.5GB safetensors → 多文件(model-00001-of-00002.safetensors等)
  → → 每文件≈6.75GB → header≈几KB → 数据≈6.75GB → header占比<0.001%
  → → 建立映射可先返回，数据访问仍可能触发page fault与存储I/O
  → → 不用只映射耗时对比完整读入；本例没有统一条件下的13.5GB计时
```

## 2. Safetensors mmap/Zero-Copy/Lazy Loading架构

```
### 2.1 mmap加载流程

safe_open("model.safetensors", framework="pt", device="cpu"):
  1. 读取8字节 → N = header长度
  2. 读取N字节 → JSON header → 解析tensor metadata
  3. mmap整个文件 → mmap(fd, filesize) → 返回虚拟内存映射
  4. → header已访问，主体数据可按需触页；不提供固定0.1ms保证
  5. 返回SafeTensorsFile对象 → 包含metadata+mmap指针

f.get_tensor("model.layers.0.weight"):
  1. 从metadata查找 → dtype=BF16, shape=[4096,4096], data_offsets=[0,33554432]
  2. 计算绝对偏移 → header_end + data_offsets[0] = 8+N+0
  3. v0.8.0现代PyTorch CPU mmap支路从私有文件storage创建相应视图
  4. → 避免先复制完整payload；不代表Python对象／header没有分配
  5. → → 页故障时才从磁盘读取 → 惰性 → 按需加载!

### 2.2 多进程共享物理内存

  同文件映射的干净页可由OS复用，但不是每个进程所有Tensor都共用同一份内存
  私有映射的写入可能产生COW私有页，转换／clone／模型参数复制会另行分配
  各rank还可能读取不同shard；设备上的参数存储不因CPU页共享而自动共享
  因而不由“8个worker、14GB文件”直接推导省98GB；需区分RSS、PSS及全作业峰值
  torch.load(mmap=True)也可映射storage，不应将所有.pt路线假定为独立eager全量复制

### 2.3 Lazy Loading选择性加载

  只加载需要的tensor → 不加载全部:
    → f.get_tensor("model.layers.0.weight") → 选取该tensor；文件映射不等同只映射一页
    → → 未请求其他Tensor不等于OS绝不预读；header也已读取，节省不按Tensor个数线性保证
    → → → 量化: 加载BF16→quantize→FP8 → 只需要target层 → 惰性quantize!

  vLLM lazy loading架构(2025新):
    → 分离weight iteration和weight materialization
    → → 1. 遍历metadata → 知道所有tensor名和shape → 不加载数据!
    → → 2. 分配GPU内存 → 根据shape和dtype → 不需要CPU数据!
    → → 3. 惰性传输 → mmap→CPU tensor→GPU tensor → 按需 → 省峰值内存!

### 2.4 Rust核心实现架构

  safetensors/src/tensor.rs → 固定v0.8.0的Rust格式实现
    → SafeTensors::read_metadata(&[u8]) → 从调用方提供的字节解析元数据
    → mmap由Python绑定等调用层组织，不将lazy_mmap当已核公开Rust API
    → SafeTensors::deserialize(&[u8]) → 解析已提供字节buffer的借用视图，不自行决定磁盘I/O
    → 序列化入口 → 组织header+data；顺序格式不自动证明固定速度优势

  Python绑定:
    → PyO3桥接 → Rust→Python → 高性能
    → Python框架helper依赖原生扩展，不据helper为.py就推定存在无Rust纯Python解析fallback
    → → safetensors/torch.py → PyTorch集成 → load_file/save_file
    → → safetensors/numpy.py → NumPy集成
    → → safetensors/tensorflow.py → TF集成
    → → safetensors/flax.py → JAX/Flax集成

  C++ header-only实现:
    → safetensors_cpp/ → 单头文件 → #include → 无依赖!
    → → vLLM/TensorRT-LLM可能用C++版本 → 高性能 → 无Python overhead

  多语言绑定:
    → Python → Rust → Go → JavaScript/WASM → 全覆盖!
    → → WASM → 浏览器加载 → 模型在Web端 → 未来方向!
```

## 3. GGUF — llama.cpp CPU/边缘推理格式

```
GGUF = GGML Universal Format → llama.cpp专用 → 消费级硬件推理!

文件结构:
  → 单文件 → 所有权重+元数据+tokenizer+量化参数 → 一站式!
  → → header → metadata(KV pairs) → tensor info → tensor data → 连续!
  → → → 不需要多个文件 → 单文件分发 → 下载简单!

量化内置:
  → Q2_K → 2bit量化 → 最小 → 精度损失大 → 极端压缩
  → Q3_K/Q4_K/Q5_K → 3/4/5bit → 平衡 → Q4_K_M最常用
  → Q6_K/Q8_0 → 6/8bit → 高精度 → 体积大 → 接近FP16
  → IQ量化(IQ2_XS/IQ3_XXS) → 2024新 → 极端压缩 → 进一步减小
  → → → 内置量化! → 不需要外部quantize → 直接下载即用!

GGUF关键特性:
  → mmap支持 → 类似safetensors → 惰性加载 → CPU内存映射
  → tokenizer内置 → 不需要额外tokenizer.json → 一文件一切!
  → 元数据丰富 → general.architecture→模型类型→注意力类型→...
  → → 单文件推理 → llama.cpp直接加载 → 无需PyTorch/safetensors依赖!

GGUF vs safetensors:
  → GGUF: 推理专用 → CPU优化 → 内置量化 → 单文件 → llama.cpp生态
  → safetensors: 命名Tensor容器，无完整计算图；可承载量化表示的数据与scales
  → → 不可互换! → safetensors→GGUF需要convert.py → 量化→格式转换!

RTX 4090 GGUF:
  → llama.cpp可有CUDA路线；文件大小不含KV／workspace，不能据4GB断言可部署六个
  → → vLLM的当前GGUF插件与限制见第8节，不再笼统写“不支持GGUF”
  → → → GGUF用于llama.cpp/ollama → safetensors用于vLLM/SGLang → 不同工具!
```

## 4. ONNX — 跨框架模型交换格式

```
ONNX = Open Neural Network Exchange → Microsoft+Meta+AWS → 跨框架标准!

文件结构:
  → 单文件 → 模型图(nodes+edges)+权重+元数据 → protobuf编码
  → → graph = 计算图 → 节点=算子 → 边=tensor流动 → 完整模型定义!
  → → → safetensors存命名Tensor而非计算图；ONNX同时表达图结构

ONNX Runtime推理后端:
  → CPU → OpenVINO/Intel → x86优化 → AVX-512/VNNI
  → GPU → CUDA/TensorRT → NVIDIA优化 → kernel融合
  → NPU → CoreML/Hexagon → 移动/边缘 → 低功耗
  → → → 多后端! → 同一ONNX模型 → 不同硬件 → 不同后端 → 自动选择!

ONNX量化:
  → 动态量化 → INT8权重 → 运行时quantize → 简单但精度损失
  → 静态量化 → INT8权重+INT8计算 → 校准数据 → 更好精度
  → QDQ(QuantizeDequantize) → 量化节点嵌入图 → 精确控制 → 2025新!
  → → → Olive工具 → PyTorch→ONNX→量化→优化 → 自动化管线!

ONNX vs safetensors:
  → ONNX: 图+权重 → 跨框架 → 多后端 → 推理优化 → 但训练不支持!
  → safetensors: 命名Tensor → 跨框架 → 可映射 → 无完整计算图
  → → → 互补! → ONNX用于硬件优化推理 → safetensors用于模型分发!
```

## 5. PyTorch pickle (.pt/.bin) — 训练checkpoint格式

```
PyTorch checkpoint = torch.save/torch.load → 对象元数据与Tensor storage → 不只用于训练

存储内容:
  → state_dict → 模型权重 → 必须存
  → optimizer.state_dict → AdamW m+v → 必须存(恢复训练!)
  → scheduler.state_dict → LR调度状态 → 建议存
  → rng_state → 随机数状态 → 建议存(恢复训练!)
  → epoch/step → 进度标记 → 建议存
  → → 是否足够恢复还取决于数据身份、采样器、版本及分布式状态，非只看.pt后缀

安全风险:
  → pickle反序列化 → 执行任意Python代码 → 恶意模型文件 = 远程代码执行!
  → → 自定义反序列化可能执行任意系统命令；此处不执行安全负例
  → → → 下载不明来源.pt文件 → 可能被攻击 → 必须信任来源!
  → → → → HuggingFace: 只信任safetensors → 自动将.bin转为safetensors → 安全!

加载性能:
  → 元数据解码与storage读取是不同成本；torch.load(mmap=True)也可按需读取storage
  → 比较时必须同样计入全数据访问、缓存、转换和设备同步，撤回旧300000倍说法

内存开销:
  → eager／mmap路线、实际触页、COW和模型复制不同，不能按后缀推导多进程常驻内存
  → 没有本页8进程PSS／峰值结果，不保留112GB对14GB的性能结论

恢复格式的取舍:
  → .pt便于表达常见训练state_dict；weights_only受限加载仍要求信任来源
  → safetensors能存优化器Tensor，但不能直接序列化任意Python对象图
  → 完整恢复需额外组织标量／结构／数据与RNG状态，格式选择不等于恢复协议
```

## 6. vLLM Weight Loading Pipeline

以下为旧概念管线，不是固定当前vLLM版本的类名／调用栈；本轮只校正映射、dtype和未测耗时，不作引擎实现验收。

```
文件: vllm/v1/worker/weight_loader.py + vllm/model_executor/model_loader/

### 6.1 加载管线架构

vLLM weight loading pipeline (discover→detect→parse→map→allocate→transfer):

Step 1: Discovery → 扫描模型目录 → 找所有权重文件
  → glob(model_dir, "*.safetensors") → 找safetensors文件
  → glob(model_dir, "*.bin") → 找pickle文件 → 优先safetensors!

Step 2: Format Detection → 识别文件格式
  → .safetensors → SafetensorsModelLoader → mmap路径
  → .bin → 具体loader的torch加载策略，不能只凭后缀判慢
  → .gguf → GGUFModelLoader → llama.cpp路径
  → → 文件格式偏好与实际加载策略须按具体版本／配置核对

Step 3: Metadata Parsing → 解析tensor元数据
  → safetensors → 读8字节header长度 → 读JSON header → tensor列表
  → → 不读数据! → 只读metadata → 惰性! → 知道所有tensor名和shape
  → → → torch checkpoint也可有懒加载，不能以此概括全部pickle路线

Step 4: Weight Mapping → 映射tensor名到模型参数
  → source_name → target_name → 可能需要rename/concat/split
  → → 例: "model.layers.0.self_attn.q_proj.weight"
  → → → → 映射到 model.self_attn.q_proj.weight → 直接对应!
  → → → → 例: QKV合并 → q/k/v各自weight → concat为一个QKV weight → 合并!
  → → → → → WeightMapping → 映射规则 → 处理rename+concat+split!

Step 5: Memory Allocation → 分配GPU tensor
  → 根据shape+dtype → torch.empty(shape, dtype, device="cuda")
  → → 不填充数据 → 只分配内存 → 快!
  → → → 设备分配成本需单独测量，不在此给固定耗时

Step 6: Weight Transfer → CPU→GPU数据传输
  → mmap→CPU tensor → GPU tensor → cudaMemcpy
  → → 惰性! → 逐tensor传输 → 不一次性全部!
  → → → 带宽、同步、页读取与转换会影响总时间；没有本页统一PCIe计时

### 6.2 SafetensorsModelLoader — mmap优化路径

class SafetensorsModelLoader:
  → load_model() → safe_open(path, framework="pt") → mmap!
  → → 遍历f.keys() → 逐tensor → f.get_tensor(name) → mmap→CPU
  → → → CPU→GPU → cudaMemcpy → 逐层加载!
  → → → → 仍创建字典／Tensor等对象，CPU payload映射不等于全流程零分配
  → → → → → 完整加载还包含触页／转换／传输／参数安装，不用映射开销推总时间

  多文件处理:
    → 大模型分多个safetensors文件 → 逐文件mmap
    → → model-00001-of-00008.safetensors → 8个文件 → 8次mmap
    → → → 文件个数、header与存储访问均有成本，不提供8次映射固定耗时
    → → → → 顺序遍历文件 → 逐文件逐tensor → 线性加载 → 简单高效!

### 6.3 量化权重加载

  FP8权重加载(2025新):
    → safetensors存BF16权重 → vLLM运行时quantize → FP8
    → → load BF16 → quantize_to_fp8 → 存到FP8 GPU tensor
    → → → safetensors已能直接表示部分FP8 dtype，无需因此宣称自定义格式
    → → → → 能否消费取决于目标引擎的量化schema、scales与kernel，不以dtype回读替代
    → → → → → 量化成本需实际测量，本页不提供1s保证

  INT4权重加载:
    → AWQ/GPTQ safetensors → 自定义dtype → 可能用U8/I8存储pack后的INT4
    → → 解pack → INT4→BF16 → 或直接INT4 kernel → 无需解pack!
    → → → vLLM: INT4 weight → fused kernel → 直接INT4 GEMM → 不解pack!

### 6.4 FSDP checkpoint → safetensors转换

  FSDP训练 → 分片checkpoint → 每rank只存1/N参数
  → → 8个分片文件 → shard_0.pt/shard_1.pt/.../shard_7.pt → pickle!
  → → → 合并 → consolidate → 全参数 → 转safetensors → 分发!

  合并流程:
    → 每rank加载自己分片 → all_gather重建完整参数 → 单进程合并!
    → → 7B ZeRO-3: 8分片×~2GB=16GB → 合并→14GB safetensors → 去冗余!
    → → → 合并成本取决于布局／I/O／通信／格式，不给统一30s估计
    → → → → 合并后永久用safetensors → 不需要再pickle!
```

## 7. 格式选择决策树

```
决策树: 根据场景选择格式

训练场景:
  → 正常训练 → PyTorch .pt → pickle → optimizer+scheduler+rng → 完整恢复
  → 分布式训练 → FSDP分片 .pt → 每rank分片 → 合并后转safetensors
  → → → 只在训练时用pickle → 分发时转safetensors!

分发场景:
  → HuggingFace Hub → safetensors → 默认! → 安全+快速+mmap
  → → 实际检查发布者提供的格式，不假设所有.bin都会自动转成safetensors
  → → → 不分发.pt → 安全风险 → 只在可信环境共享!

推理场景:
  → GPU推理(vLLM/SGLang) → 检查safetensors与对应loader／量化schema支持
  → → 量化推理 → safetensors(BF16) → 运行时quantize → 或量化safetensors
  → CPU推理(llama.cpp) → GGUF是常见工具链选择，不在此评定普遍最优
  → 跨硬件推理 → ONNX → 多后端 → 硬件加速!
  → NVIDIA GPU → TensorRT-LLM等引擎按模型／部署约束评估

RTX 4090决策:
  → 训练 → 根据训练框架选择完整checkpoint，不由GPU型号决定唯一格式
  → 推理 → safetensors与受支持loader是一个选项，不承诺14GB加载时延
  → → INT4推理 → 还需匹配AWQ/GPTQ表示与kernel，不由文件后缀保证更快
  → CPU推理 → GGUF(Q4_K_M) → llama.cpp → 4GB→RTX 4090 CUDA → 可行但不如vLLM!
  → → → 本页没有同模型同条件比较，不能选出RTX 4090唯一最优方案
```

## 8. RTX 4090 Weight Format Implications

本页没有RTX 4090加载benchmark，不能由GPU型号推导固定1.2秒、26倍加速或唯一最优格式。参数文件大小也不包含KV cache、运行时workspace和模型复制，不能用24GB除以4GB断言可同时部署六个模型。

safetensors的FP8存储能力与4090的算子／量化支持是不同层次；本轮E4M3 CPU字节回读不验证推理。原文“vLLM不支持GGUF”也撤回：[vLLM官方GGUF说明](https://docs.vllm.ai/en/latest/features/quantization/gguf/)提供实验性支持入口，目前要求另装vllm-gguf-plugin，并提示优化与兼容边界；这不是本项目模型实测或4090支持认证。

若未来已有获准加载任务，需要固定模型及其量化schema、引擎与版本、实际文件／缓存状态、设备和完整测量终点，再谈格式对加载／推理的收益。本轮没有新增硬件实验，不部署模型或排M5队列。

## 参考文献

```
1. Safetensors源码和规范:
   - github.com/huggingface/safetensors — Rust core + 多语言bindings
   - huggingface.co/docs/safetensors/en/spec — 二进制格式规范
   - huggingface.co/blog/safetensors-the-new-way-to-save-tensors — 为什么safetensors

2. GGUF格式:
   - github.com/ggerganov/gguf — GGUF规范
   - github.com/ggerganov/llama.cpp — llama.cpp实现

3. ONNX:
   - onnx.ai — ONNX规范
   - github.com/microsoft/onnxruntime — ONNX Runtime

4. vLLM Weight Loading:
   - vllm/v1/worker/weight_loader.py — vLLM V1 weight loader
   - vllm/model_executor/model_loader/ — ModelLoader hierarchy

5. 我们的笔记:
   - checkpoint.md → checkpoint基础
   - vllm-weight-loading-reading.md → vLLM weight loading详解
   - quantization.md → 量化基础
   - fp8-quantization.md → FP8量化深度
