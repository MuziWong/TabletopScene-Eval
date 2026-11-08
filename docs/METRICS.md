# 指标与评测协议

本项目只读取 AgenTable 标准 `scene_export`：成功的 `result.json`、canonical 布局和对象 GLB。所有方法共享同一输入图、参考 mask、相机、照明和模型预处理。源包只读，输出写入独立实验目录。

## 渲染、归一化与前景

视觉归一化对每个方法执行一次共同相似变换：

1. 从桌子世界网格求 XZ 最大跨度，定义为桌径。
2. 将桌子 XZ 包围盒中点和最高 Y 定义的桌面中心移到原点。
3. 将桌径缩放到 1；桌子、所有对象和相对位移使用同一比例。

它保留物体与桌子的尺度比及相对布局，有意去掉全局桌径和平移差异。视觉分数因此不衡量绝对尺寸；该操作不是实际尺寸标定。物理始终读取归一化前的原始几何和初始位姿。

归一化后，联合所有方法的桌子顶点，以联合完整包围盒中心为目标，用 `shared_table_projection` 求一套共同相机距离。距离同时满足所有候选的逐顶点针孔投影约束，margin=1.08；不按各方法的目标物体独立拟合。对象超出桌子范围时不保证入画，近/远裁剪仍需覆盖场景。

默认 16 个候选为光轴／轨道仰角 `[25,35]°` × 方位角 `[0,45,90,135,180,225,270,315]°`。GPT anchor 固定为 `e01_a00_d00`，即 35°/0°。瞄准整桌中点时，光轴角与相机看向桌面中心的实际俯角不同，具体角度由运行相机坐标决定。

Blender 使用源 GLB 材质，Cycles 32 samples、降噪、白色 World 强度 0.6、三盏 AREA 灯 400/200/100 W（size=3 m），颜色管理 AgX / look=None / exposure=0 / gamma=1。渲染无仿真步，实际设备、相机、灯光、资产哈希和位姿检查留存于产物。

参考 mask 在所有方法间共享；预测 mask 来自各自渲染的可见对象 `IndexOB` pass。默认包含桌子和对象，供 GPT 同时判断支撑关系。仅在其它显式协议中设置 `foreground.include_table=false` 时，参考 mask 也须排除桌子，但桌子仍参与遮挡；该设置不适用于当前 GPT `fixed_and_best` 模式。每张图各用自己的 mask 合成统一白底，不裁剪、居中或独立缩放，因此保留额外对象及错位。标准图像指标作用于合成后的完整画布，并非把特征距离逐像素按 mask 归一化。

`--auto-mask` 只合并输入分析中已有的实例 mask，跳过 subscene 并尊重桌子选项。要求分割原图与共享参考图 SHA256 一致；记录自动来源，不当作人工 GT。

## LPIPS、DINOv2、CLIP

| 指标 | 实现与计算 | 方向 |
| --- | --- | --- |
| LPIPS | 官方 `lpips`，AlexNet；RGB 映射到 [-1,1]，比较多层归一化特征并使用学习到的通道校准、空间平均与层求和 | 越小越好 |
| DINOv2 | `facebook/dinov2-base`，ViT-B/14 最终 768 维 CLS 特征，计算两图余弦 | 越大越好 |
| CLIP | `openai/clip-vit-base-patch32`，512 维图像投影特征，计算图像之间的余弦；不使用文本或 softmax | 越大越好 |

余弦为 `dot(a,b)/(norm(a)*norm(b))`，理论范围 [-1,1]，不是准确率或概率。LPIPS 相同图应接近 0，没有严格的 1 上界；当前是冻结模型前向计算，不用于优化场景。

DINOv2 processor 将短边 resize 到 256，再中心裁剪 224×224，使用 ImageNet 归一化；CLIP 短边 resize 到 224、中心裁剪 224×224，使用 CLIP 归一化。中心裁剪可能丢失边缘物体。LPIPS 默认要求参考与渲染同尺寸，只有显式 `image_size` 才同时缩放双方。模型 revision 和实际 processor 写入结果。

每个场景、每个方法分别从 16 个共同候选取 LPIPS 最小值、DINOv2 最大值、CLIP 最大值，三项允许选择不同图片；随后对场景等权平均。GPT 默认另按一个预先配置的 `best_view_metric` 选择 best，并与固定 anchor 一起展示，不把三项各自最佳图全部发送。

实现与配置分别位于 `metrics/lpips/`、`metrics/dinov2/`、`metrics/clip/`。底层模型来源：[LPIPS](https://github.com/richzhang/PerceptualSimilarity)、[DINOv2](https://github.com/facebookresearch/dinov2)、[CLIP](https://github.com/openai/CLIP)。

## GPT：VF、IA、PP、Avg.、OR

实现、配置与完整 prompt 位于 [`metrics/gpt_evaluation/`](../metrics/gpt_evaluation/)。每个场景一次请求，将真实候选匿名编号后共同呈现，不添加虚构 baseline；顺序由 `order_seed` 控制，它不控制远端推理随机性。

| 项目 | 含义 |
| --- | --- |
| VF | 综合固定图与 best 图，判断可见形状、材质/纹理与渲染缺陷 |
| IA | 主要依据 best 图，对比输入的类别、数量、尺寸、姿态、布局与外观一致性 |
| PP | 综合两图判断可见支撑、接触、漂浮或穿插；不推测隐藏几何或动力学 |
| Avg. | 本地计算 `(VF + IA + PP) / 3` |
| OR | 多方法时模型直接给出的整体排名，1 为最好；不由 Avg. 排序生成 |

VF/IA/PP 取 1–7 分，允许小数；prompt 给出 1、4、7 的锚点，中间质量插值，不强迫最好候选得 7。多方法 OR 是无并列的整数排列 `1..M`，跨场景求平均名次，参评集合不同不能直接比较。

**单方法仍评 VF/IA/PP 和 Avg.，OR 必须为整数 1，只有占位约定，没有方法间比较意义。** prompt 明确禁止编造额外候选；审计记录 `candidate_count=1`、`ranking_comparative=false`。

当前默认使用**统一视觉归一化后的单组前景图，保留桌子**：

1. 参考前景图只发送一次。
2. 每个方法发送固定图：`e01_a00_d00`，0° 方位角／35° 光轴仰角。
3. 同一方法再发送 best 图：从相同 16 个候选中，按实验前配置的单一图像指标选择。

总图数为 `1+2M`，其中 M 是方法数；两方法共 5 张，单方法共 3 张。固定与 best 可能选到同一视角，但两张图始终属于同一方法，不作为两个候选重复打分。每方法仅返回一套 VF/IA/PP/Avg. 与一个 OR。

默认设置在 `metrics.gpt_evaluation.options`：

```json
{
  "view_selection": "fixed_and_best",
  "best_view_metric": "LPIPS"
}
```

`LPIPS` 取候选最小值；也支持预先指定 `DINOv2` 或 `CLIP` 取最大值。所有方法采用同一个选择指标，不能看到评分后为不同方法切换规则。选择指标、视角 ID 和图片哈希留存在本地审计，**不把选择分数发送给 GPT**，避免数值锚定。IA 主要根据 best 与参考图的对齐；VF/PP 综合固定与 best 的可见证据。模型不读取网格或 FCL 结果，PP 仍不能证明隐藏部分不存在碰撞，也不能代替物理检测。

`fixed_and_best` 要求 `metrics.input_view.enabled=true`，其 `view_selection` 为 `best_per_metric` 或 `mean`，且计算项包含 `best_view_metric`；同时要求 `foreground.enabled=true`、`foreground.include_table=true`。即使图像汇总采用 `mean`，GPT 的 best 仍按该指标的逐候选分数选择。需要旧的单固定图路径时可显式设置 GPT `view_selection="fixed"`；旧的前景／原始支撑双组呈现属于历史协议，不是当前默认。新旧图像呈现或 prompt 的分数必须分开记录。

默认通过 OpenAI 兼容 Chat Completions API，`image_detail=high`、JSON mode；本地严格检查候选、字段、范围和排名。JSON mode 本身不保证业务 schema。实际 prompt、源/请求 prompt 哈希、候选映射、图片哈希、模型、原始响应与用量可留存审计；改 prompt 或模型应重新评测，不能混用旧分数。

指标名称、1–7 量表及整体排名参考 [TabletopGen](https://arxiv.org/html/2512.01204v4#S4.SS1) 和 [ArticuTable](https://arxiv.org/html/2610.05249v1#A4.SS1)。[SceneEval 的公开 prompts](https://github.com/3dlg-hcvc/SceneEval/blob/116881e7945dcf7dbc58f7c64590a4c0b99c5cf5/vlm/prompts.yaml) 与 [SceneSmith critic](https://github.com/nepfaff/scenesmith/blob/67cc408fd38334b4a926efef45e284302ed5055b/scenesmith/prompts/data/furniture_agent/stateful_critic_agent.yaml) 提供了结构化判据的参考；本项目 prompt、fixed+best 呈现、归一化和 16 视角协议是独立设计，未声称它们是原论文的原版 judge 或复现了论文绝对分数。

## ColO、ColS 与五档阈值

实现与配置位于 `metrics/physical_validity/`。只检查归一化前初始位姿中的不同非桌对象，不移动物体、不执行落稳或仿真。

闭合凸网格直接使用；其它网格用 V-HACD 构造凸分解近似，默认最多 128 凸块、每块最多 64 顶点、resolution=400000。GLB 节点变换先烘焙，场景变换作用到凸块；网格来源、原尺度、分解参数与缓存哈希保留。不会用单个整体凸包填平非凸对象的所有空隙。

AABB 预筛后对候选凸块对执行 FCL Convex 查询，采用 LIBCCD/MPR 穿透估计。对象对的深度取所有凸块对估计的最大值。它不是完整非凸实体的精确最小平移距离，凸分解和求解器均可能有近似误差。

一次分解和完整深度查询后，复用为五档：

```json
{
  "min_penetration_m": 0.001,
  "numerical_tolerance_m": 1e-9,
  "penetration_thresholds_m": [0, 0.0001, 0.0003, 0.0005, 0.001]
}
```

即 0、0.1、0.3、0.5、1 mm，统一严格条件 `depth > threshold + 1e-9 m`。这里的数值 guard 不代表求解器误差有 1e-9 m 的保证。导出尺度若未经过真实标定，“毫米”仅为导出名义米坐标换算。

场景 s 有 n 个非桌对象，碰撞对象对数为 c：

- `ColO_s = 100 × c / C(n,2)`，每对只计一次。
- `ColS_s = 100` 当 c>0，否则为 0。

至少需要两个非桌对象。跨场景先计算场景内比例，再等权平均 ColO；平均二值 ColS 就是存在碰撞的场景百分比，不能按全数据集对象对混池。主 `summary` 使用 1 mm；额外五档保存在场景 `details[方法].physical_validity.threshold_sweep` 和汇总 `physical_validity_threshold_sweep`，保留每对判定、对象对数与样本数。

排除桌子意味着这些指标不检查穿桌；它们也不检查漂浮、支撑、动态稳定性或关节运动。ColO/ColS 为 0 只代表当前近似几何与阈值下未检出非桌对象间相交，不代表场景物理有效。
