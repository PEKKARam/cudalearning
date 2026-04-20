# 阶段A：CUDA 基础算子（第 1 课）

本课实现你的第一个 CUDA 算子：向量加法（Vector Add）。

## 学习目标

- 理解 CUDA 执行模型的基础概念：
  - `threadIdx`、`blockIdx`、`blockDim`、`gridDim`
- 启动并运行一个基础 Kernel
- 对比 GPU 结果与 CPU 参考结果
- 使用 CUDA Event 测量 Kernel 执行时间

## 文件说明

- `vector_add.cu`：包含较详细中文注释的教学示例

## 编译

如果当前终端里找不到 `cl.exe`，请显式指定 `-ccbin`：

```powershell
nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" .\stageA_cuda_basics\vector_add.cu -o .\stageA_cuda_basics\vector_add.exe
```

如果 `cl.exe` 已在 PATH 中（例如 VS Native Tools 终端）：

```powershell
nvcc .\stageA_cuda_basics\vector_add.cu -o .\stageA_cuda_basics\vector_add.exe
```

## 运行

```powershell
.\stageA_cuda_basics\vector_add.exe
```

预期关键输出：

- `Result check: PASS`
- Kernel 执行时间（毫秒）
- 近似吞吐率（GB/s）

## 建议的下一步练习（仍属于阶段A）

1. 将 `n` 从 `1<<20` 改为 `1<<24`，比较运行时间变化。
2. 尝试 `threads_per_block` = 128、256、512，比较吞吐率变化。
3. 去掉 grid-stride loop，观察其对可扩展性的影响。
4. 在正式计时前增加一次 warm-up 启动，观察时间稳定性。
