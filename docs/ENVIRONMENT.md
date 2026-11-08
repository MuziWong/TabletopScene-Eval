# 环境与 API

## 已有环境与首次安装

本机已创建并由 Conda 注册独立环境 `tabletopscene-eval`：

| 项目 | 已验证状态 |
| --- | --- |
| 路径 | `/home/dex/miniconda3/envs/tabletopscene-eval` |
| Python | 3.11.18 |
| torch / torchvision | 2.7.1+cpu / 0.22.1+cpu |
| Conda 登记、环境元数据 | 已确认 |
| 核心依赖导入、`pip check` | 通过 |
| FCL 实际相交/分离查询 | 通过 |

本机从项目根目录激活并登记源码即可：

```bash
conda activate tabletopscene-eval
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

新机器首次安装时，先安装 Conda，再从项目根目录执行；已有同名环境时直接激活，不重复创建：

```bash
conda env create -f environment.yml
conda activate tabletopscene-eval
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

创建环境时 Conda 自动登记名称，不需要注册 Jupyter kernel。可用以下命令核验：

```bash
conda env list
conda run -n tabletopscene-eval python -c "import sys; print(sys.executable, sys.version)"
conda run -n tabletopscene-eval python -m pip check
```

[`environment.yml`](../environment.yml) 提供已验证的 Linux x86_64 / Python 3.11 CPU 安装组合，管理 Python 与固定版本依赖；其它平台需确认对应二进制包可用。[`pyproject.toml`](../pyproject.toml) 管理项目安装和命令入口。先完成环境文件的依赖安装，再使用 `--no-deps` 登记源码；`--no-build-isolation` 使用环境中已有的 setuptools（项目要求 ≥68）。核心依赖为 PyTorch/torchvision、NumPy/SciPy/Pillow、LPIPS、Transformers、OpenAI SDK、python-dotenv、Trimesh、python-fcl、vhacdx。物理使用 CPU，不需要仿真器、DexSim、Drake 或 `rtree`。

## 外部 Blender

主 Conda 不安装 `bpy`。框架通过配置中的 `rendering.executable` 调用外部 Blender，由 Blender 自带 Python 执行渲染：

- 本机程序：`/snap/bin/blender`，已验证 Blender 4.5.15 LTS。
- 内置 Python：3.11.15，与主 Conda 独立。
- 已验证 GPU：RTX 5060 Ti，Cycles OPTIX。
- 可显式选择 `device: "OPTIX"`、`"CUDA"` 或 `"CPU"`。

其它机器在 [`configs/render.json`](../configs/render.json) 中修改程序路径与设备。请求的 GPU 后端不可用时明确报错，不静默回退 CPU。主环境的 CPU 版 PyTorch不会阻止外部 Blender 使用 GPU；图像特征仍在主环境计算。需要 PyTorch CUDA 加速时应成对更换 torch/torchvision，并重新记录环境。

只运行 `--physics-only` 时无需安装或启动 Blender。预训练权重首次使用可能下载；LPIPS 使用 PyTorch 缓存，DINOv2/CLIP 使用 Hugging Face 缓存。准备好缓存后可以离线计算图像指标，GPT 仍需要网络。

## GPT API

已有私有 `.env` 时直接使用；首次配置可执行：

```bash
cp -n .env.example .env
# 编辑 .env，填写自己的凭证和模型。
tabletopscene-eval --check-api --env-file .env
```

支持 `OPENAI_API_KEY`，以及可选的 `OPENAI_BASE_URL`、`OPENAI_MODEL`。[`metrics/gpt_evaluation/config.json`](../metrics/gpt_evaluation/config.json) 显式加载项目根目录 `.env`。配置中显式的 model/base_url 优先于环境值；已有进程环境变量优先于 `.env` 同名变量。凭证不写入 JSON 配置、命令行或评测审计记录。

连通性检查只发送小型文本 JSON 请求，不发送场景图，不能替代图像评测验证；也可通过 `python -m tabletopscene_eval --check-api --env-file .env` 调用。默认评测会调用 GPT；使用 `--no-gpt` 或 `--physics-only` 可关闭。一次只有一个方法时仍评分 VF/IA/PP，OR 固定约定为 1，不构造虚假的 baseline。

## 验证与记录

从项目根目录运行离线测试：

```bash
python -m unittest discover -s tests -v
```

普通测试不调用付费 GPT。真实模型与 Blender 测试按显式开关运行：

```bash
TABLETOP_EVAL_REAL_MODELS=1 python -m unittest tests.test_real_input_view -v
TABLETOPSCENE_TEST_BLENDER_RENDER=1 python -m unittest tests.test_blender_backend -v
```

保留每次实验的配置、模型与 prompt 哈希、源资产哈希，以及 Blender/驱动版本。需要导出完整环境时使用：

```bash
python -m pip freeze > experiment-requirements.txt
conda env export > experiment-environment.yml
```

包版本固定不保证不同设备逐位一致，也不保证远端 GPT 重复评分完全相同。所有算法与协议边界见 [METRICS.md](METRICS.md)。
