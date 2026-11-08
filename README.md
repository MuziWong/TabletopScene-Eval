# TabletopScene-Eval

读取 AgenTable 格式的 `scene_export`，统一评测单个结果或整批场景。保留一套主流程：**Blender 渲染 → 前景图像指标与 GPT 评分；原尺度网格 → V-HACD/FCL 五档碰撞指标**。

## 环境

本机已创建并注册独立 Conda 环境 `tabletopscene-eval`，无需重新创建：

```bash
conda activate tabletopscene-eval
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
```

新机器先执行 `conda env create -f environment.yml`，再执行上述命令。Blender 是外部程序，本机路径 `/snap/bin/blender`；主 Conda 无需安装 `bpy`、DexSim 或 Drake。安装与 API 配置见 [ENVIRONMENT.md](docs/ENVIRONMENT.md)。

GPT 使用根目录私有 `.env` 中的 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`、`OPENAI_MODEL`；已有文件无需覆盖。添加 `--no-gpt` 可关闭 API 评测。

## 单个与批量评测

单个导出包，显式使用已有自动分割生成参考前景：

```bash
tabletopscene-eval /path/to/scene_export --auto-mask --output runs/one
```

也可以提供自己准备的图像与 mask：

```bash
tabletopscene-eval /path/to/scene_export \
  --reference /path/to/input.png --mask /path/to/mask.png --output runs/one-manual
```

批量递归寻找 `scene_export/result.json`：

```bash
tabletopscene-eval /path/to/outputs --batch \
  --reference-root /path/to/images --mask-root /path/to/masks --output runs/batch
```

场景 ID 为相对于输入根的 case 路径。例如 `outputs/2/02-01/scene_export` 对应 `2/02-01`，默认匹配 `images/2/02-01.png` 和 `masks/2/02-01.png`。其它扩展名可用 `--reference-pattern '{scene_id}.*'`、`--mask-pattern '{scene_id}.*'`；匹配不到或匹配多份都会报错。

比较多个方法时，用重复的 `--method NAME=ROOT` 替代位置参数；各根目录应具有相同场景 ID：

```bash
tabletopscene-eval --method AgenTable=/data/agentable/outputs --method Other=/data/other/outputs \
  --batch --reference-root /data/images --mask-root /data/masks --output runs/compare
```

输入只接受标准 `scene_export` 包，也可指定单个 case 或其 `result.json`。原图默认查找 case 下的 `input_analysis/success/original.png`，否则需提供 `--reference` 或 `--reference-root`。前景 mask 必须显式提供，或通过 `--auto-mask` 合并已有 `scene_graph.json` 中的实例 mask；自动 mask **不是人工 GT**，正式实验应核验。默认前景包含桌子和物体；mask 须与原图同尺寸、白色为前景、黑色为背景。

常用选项：

| 选项 | 作用 |
| --- | --- |
| `--config configs/default.json` | 指定评测协议；默认已加载此文件 |
| `--no-gpt` | 关闭远端 GPT 调用 |
| `--physics-only` | 只评原尺度网格，不需要图像、mask、Blender 或 API |
| `--continue-on-error` | 批量记录失败并继续；有失败仍返回非零退出码 |

默认严格检查全部方法的同名场景；不完整的方法组合不会进入汇总。输出目录须为新实验目录，已有 `summary.json` 或 `results.json` 时拒绝复用，不覆盖或断点续跑。也可用 `python -m tabletopscene_eval` 调用相同入口。

## 配置与结果

每项 metric 的实现与配置放在自己的目录：

```text
metrics/
├── lpips/               # 感知距离；metric.py + config.json
├── dinov2/              # 视觉特征余弦；metric.py + config.json
├── clip/                # 图像特征余弦；metric.py + config.json
├── gpt_evaluation/      # VF/IA/PP、Avg./OR；另含 prompt.txt
└── physical_validity/   # V-HACD/FCL 与五档阈值
```

外层 `configs/` 只组织共享的 `default.json`、`camera.json`、`render.json` 和覆盖示例。使用 `extends` 组合片段；对象递归合并、数组整体替换，路径相对于声明它的配置文件。修改 metric 参数时编辑对应目录的 `config.json`，或从实验配置显式覆盖。

默认视觉协议将各方法桌面中心移至原点、桌径统一为 1，整场景使用同一个等比例变换。所有方法共享 16 个相机（25°/35° × 8 方位角）、Cycles 32 samples 和统一棚拍光照。LPIPS/DINOv2/CLIP 分别选最佳视角；GPT 使用预先固定的 35° anchor。单方法也可评 GPT，**OR=1 仅为约定，没有方法间排名意义**。

物理始终使用归一化前的原始网格与初始位姿，排除桌子，不运行仿真。一次分解和深度查询，同时计算 **0、0.1、0.3、0.5、1 mm**；主 ColO/ColS 仍取 1 mm。算法、公式、prompt 来源与限制见 [METRICS.md](docs/METRICS.md)。

结果保存在 `summary.json`；逐场景完整结果位于 `scenes/<场景ID>/result.json`，同时保留发现记录、配置、渲染、mask 与 GPT 审计信息。五档汇总位于 `physical_validity_threshold_sweep`。失败场景单独记录，不作为零分混入均值；成功场景等权汇总。
