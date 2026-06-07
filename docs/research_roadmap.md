# PointCraft 研究路线图

> 给自己看的导航文档：要走的方向、该读的论文、该参考的技术、该试的方向，
> 以及每个环节在哪儿能接触/对接原田（Harada-Kurose-Mukuta）实验室的研究成果。
> 不是套磁材料——是一条能持续走下去的研究主线。

---

## 0. 一句话主张

**Aerial-to-Embodied Semantic Scene Completion**：
从航测稀疏点云（自上而下，主要只看得到屋顶/地面/植被顶）出发，
学习补全出城市的**完整体素占据 + 语义**，以 PLATEAU LOD2 / CityGML 作为监督真值，
并在 Minecraft 这样的离散体素世界里实例化为可交互的具身环境。

- **输入**：航测点云 → 体素化的部分占据（+ 高度/法向/强度/fTSDF 特征）
- **输出**：每体素 {占据∈{0,1}, 语义∈{建筑/屋顶/立面/植被/地面/道路/…}}
- **监督**：LOD2 灌成体素真值（建筑完整体量）+ 点云逐点类别
- **指标**：完成 IoU、语义 mIoU、**未观测区（立面/体量）补全准确率**、MC 可视化

**为什么这个角度好**：航测"只拍得到屋顶"不是缺陷，而正是
"limited-observation 三维重建"这一研究命题本身——这恰是原田实验室官网明确写的方向。

---

## 1. 核心命题拆成两条已有研究线

| 线 | 任务名 | 成熟度 | 我们的关系 |
|---|---|---|---|
| A | **Semantic Scene Completion (SSC) / 3D 语义占据补全** | 成熟，有 benchmark | 直接做 backbone；但现有都是车载街景，**航测/城市尺度是空白** |
| B | **航测 LiDAR → LoD2 建筑重建** | 活跃，较拥挤 | 最接近的前作（Point2Building）；用"体素+语义+具身输出"与之错开 |

**差异化（让工作有独立价值的关键）**：
1. 输出**体素 + 语义**而非多边形网格 → 天然接具身环境 / Minecraft。
2. **整场景补全**（建筑+植被+地面+道路语义），而非仅建筑外壳 mesh。
3. 聚焦"屋顶可见、立面/体量不可见"的**生成式补全** + 城市尺度 + 具身实例化。

---

## 2. 阅读路线（按依赖顺序，每篇标注「拿什么」）

### 阶段 1 — 基础：稀疏 3D 表示与卷积
- **SparseConvNet / Submanifold Sparse Conv**（Graham 2018）— 稀疏卷积的根。
- **Minkowski Engine**（Choy, CVPR 2019）— 我们 backbone 的候选框架。拿：稀疏张量 + 广义卷积。
- **SECOND / spconv**（Yan 2018）— 工程上更常用的稀疏卷积库。拿：实际代码栈。
- *可选* TorchSparse（更快的库）。

### 阶段 2 — SSC 主线（这是「重活」的核心谱系）
- **SSCNet**（Song, CVPR 2017）— 任务开山，占据+语义联合预测的定义。
- **LMSCNet**（2020）— 轻量 2D/3D 混合，理解 BEV 思路。
- **S3CNet**（CoRL 2020）— **架构思路参考，非代码模板**（已更正 2026-06）：BEV 2D 分支 +
  几何感知损失值得借鉴，但**无可靠官方代码**，不要当骨架克隆。
- **SCPNet（CVPR'23，spconv+Cylinder3D）/ JS3C-Net（AAAI'21，spconv1.0）** ⭐ —
  **M2 实际的 spconv 代码参考**（读结构：编码-解码、稀疏张量管线、loss）。两者都钉死在
  **旧 spconv/CUDA**，只读设计、别指望在本机栈上直接跑。**SSC-RS** 等后续 SOTA 理解改进点
  （语义分割辅助、多尺度、补全先验）。
- **dual-path SSC**（2025, SemanticKITTI 62.6% IoU）— 当前 SOTA，对照基线。
- *综述* "Semantic Scene Completion: A Survey" — 快速建立全局地图。

### 阶段 3 — 3D Occupancy（更新的同源框架，自动驾驶圈）
- **MonoScene / VoxFormer / SurroundOcc / OccNet** — occupancy prediction 的现代 transformer 化做法。拿：query-based 占据预测、可借的 decoder 设计。
- **OpenOccupancy** benchmark — 评测协议参考。

### 阶段 4 — 航测建筑重建（最接近的前作，B 线）
- **PolyGen**（Nash, ICML 2020）— 自回归生成网格（顶点→面）的祖先。
- **Point2Building**（2024）⭐ — **最接近的前作**：航测 LiDAR → LoD2 mesh，稀疏 CNN + 自回归 transformer，用 Zurich/Berlin/Tallinn 的 LiDAR↔LoD2 配对训练。拿：①数据配对可行性证明 ②生成式从缺失观测推断的思路。**这是主要竞争者，必须读透并写清差异。**
- **City3D / Points2Poly / PolyFit** — 传统几何派（平面拟合+优化）。拿：对照、传统 baseline。
- **ISPRS 2025 LoD2 benchmark**（aerial lidar + footprints）— 现成评测/数据。
- **BuildingWorld**（2025）— 城市基础模型用的结构化建筑数据集。拿：预训练/对照数据。

### 阶段 5 — 生成式补全（把"补出未观测部分"做实）
- 点云补全：**PCN / PoinTr / SeedFormer** — shape completion 思路。
- *进阶* 3D diffusion / latent diffusion for shape & scene completion — 作为 stretch 方向。

### 阶段 6 — 具身 / Minecraft 输出端
- **MineDojo / MineRL / Voyager / Plan4MC** — Minecraft 作为具身 AI 平台的研究线。拿：把重建结果接成"可交互环境"的论证。
- **Habitat**（Savva 2019）— 具身 AI 平台范式参考。

---

## 3. 与原田实验室成果的接触点 ★（你的主要关注目标）

按环节标出"在哪儿能用到 / 对接他们的研究"，这决定了你工作的哪些部分能与他们对话：

| 环节 | 对接的原田成果 | 怎么接触 |
|---|---|---|
| 体素表示（概念，非代码） | **GeoSVR**（NeurIPS'25 Spotlight；image→可微渲染、**逐场景优化**的表面重建）— ⚠ **非监督式补全、代码不可复用**（已更正 2026-06） | 仅作**概念参照**（"显式稀疏体素能得到准确完整几何"）+ 同实验室活跃线（Lin Gu 等），保持关注、不导入 |
| 整个"受限观测重建"命题 ★ | 实验室 **Future Directions** 原话："**limited observation**"、"**estimating unobservable areas**"、"**a wide range of areas such as towns and suburbs**"、"**future prediction … by accumulating data**" | **direction-fit，非 method-identity**：PointCraft 的城市设定是该"广域"方向的一个**实例**。注意其原文说 **towns/suburbs**，**未**说"city/urban"——不要声称他们呼吁"城市尺度" |
| ~~重建/修复任务定位~~ | ~~CVPR'26 3DRR Challenge~~ — ⚠ **已更正 2026-06**：该届 3DRR 是**低光/烟雾退化**复原，**不是**航测立面补全，**不作为本项目的目标/投稿 venue** | 对齐的是实验室**主方向**，不是这个 challenge |
| 点云配准 / 部分匹配 | **Lepard**、**Neural Deformation Pyramid** | 多 tile / 多时相对齐时直接用 |
| 神经渲染（外观/材质 stretch） | **NeRF 系**、**Luminance-GS / Aleth-NeRF / I²-NeRF**、3DGS | 若做立面外观补全，参考他们的渲染方法 |
| 具身 / 下游 | **Cross-Embodiment Offline RL**、scene-graph grounding（SceneProp） | 把重建环境接成具身 agent 任务时对接 |
| 语义场景理解 | DEJIMA（日语图像-VQA 数据集）、scene-graph 工作 | 语义层级设计、未来 vision-language 扩展参考 |

**项目定位（更正 2026-06）**：不再表述为"静态城市补全 + 借 GeoSVR 代码"。诚实表述为：
**在受限观测下、由理解驱动的补全**，产出**可具身化、带身份语义的环境**；具身 / 世界模型
是**未来工作愿景**（例如与实验室 R2-Dreamer 线对齐），**非当前交付物**。与实验室的关系是
**方向契合（direction-fit）**——其 Future-Directions 的 "limited observation / estimating
unobservable areas / towns and suburbs / future prediction by accumulating data" 正是本
命题的上位方向；GeoSVR 仅为"显式稀疏体素可得完整几何"的**概念佐证**，不是可复用方法。

---

## 4. 技术栈（工程部分走捷径，不造轮子）

- **点云 IO / 处理**：laspy、PDAL、Open3D
- **CityGML / LOD2 解析**：现有 OBJ/MTL 解析（已写）、citygml4j（如需原始 CityGML）
- **稀疏卷积 backbone**：spconv（首选）或 MinkowskiEngine
- **MC 读写**：mcschematic（已用）、Amulet-Core、litemapy —— 绝不自己写世界格式
- **程序化城建参考**：Arnis（OSM→MC）、Build The Earth —— 仅借鉴 IO/工作流
- **数据集**：PLATEAU（主力，日本 LiDAR↔LoD2 配对）；SemanticKITTI（SSC 预训练/对照）；SensatUrban / DALES / Toronto-3D（城市/航测点云分割）；ISPRS Vaihingen、BuildingWorld

---

## 5. 里程碑（每步明确产出 + 是否触达原田成果）

- **M0 数据配对**（工程，快）：LiDAR 点云 ↔ LOD2 做成体素训练对（输入=部分占据，目标=完整占据+语义）。复用现有对齐代码。➜ 产出：可训练数据集。
- **M1 确定性 baseline**（已基本完成）：现在的栅格化管线，作为对照下限。
- **M2 学习式占据补全**（研究核心起步）：spconv UNet 从部分占据预测完整占据。第一步=**单 tile 过拟合** `09LD1874`，须明显超过 M1 下限（strict 未观测 IoU 0.061）。指标：完成 IoU + 未观测 IoU（复用 `pointcraft.metrics` 三档 cutoff）。➜ *代码参考*：SCPNet/JS3C-Net（spconv，读结构）；*概念*：GeoSVR 显式体素。
- **M3 + 语义双头**：加语义类别预测。指标：语义 mIoU。
- **M4 立面/体量生成式补全**：专门评测"未观测区"补全质量（这是与 Point2Building 差异化、与原田"受限观测重建"对齐的核心卖点）。➜ *接触点*：limited-observation reconstruction、生成式补全。
- **M5 具身 / MC 演示**：体素→Minecraft，可交互环境 demo。➜ *接触点*：MineDojo/具身线、cross-embodiment。

---

## 6. 风险与取舍

- **新颖性被 Point2Building 稀释** → 用「体素+语义+具身输出+生成式立面补全」组合切口区分；强调下游具身可用性，而非纯几何精度比拼。
- **LOD2 立面是平直外壳，不含真实窗/材质** → 真值精确到"体量+语义层级"即可；外观/窗格作为可选 stretch（点云/影像弱监督 + 神经渲染）。
- **数据规模 / 算力** → 先单城多 tile 跑通 M2/M3，再谈规模与多城泛化。
- **别陷进工程** → 上色、格式、游戏内搭建一律走现成库；精力投在 M2–M4 的学习模型上。

---

## 7. 立即可做的第一步

进入 **M0**：把当前 `pointcraft` 管线产出的体素网格 + LOD2 体量真值，
固化成 `(部分占据, 完整占据+语义)` 的训练样本（按 tile 切块、存 `.npz`/稀疏张量）。
这一步复用现有对齐与体素化代码，是后面所有学习实验的地基。

---

## 附：关键文献清单（可直接检索）

- Choy et al., *Minkowski Engine* (CVPR 2019)
- Graham et al., *Submanifold Sparse Convolutional Networks* (2018)
- Song et al., *SSCNet: Semantic Scene Completion from a Single Depth Image* (CVPR 2017)
- *S3CNet: A Sparse Semantic Scene Completion Network for LiDAR* (CoRL 2020) — arXiv:2012.09242
- *Dual-path network for semantic scene completion of single-frame LiDAR* (2025)
- Nash et al., *PolyGen* (ICML 2020)
- *Point2Building: Reconstructing buildings from airborne LiDAR* (2024) — arXiv:2403.02136
- *GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction* (NeurIPS'25 Spotlight) — arXiv:2509.18090 — github.com/Fictionarry/GeoSVR
- *ISPRS 2025: benchmark on LoD2 building reconstruction from aerial lidar and footprints*
- *BuildingWorld: A Structured 3D Building Dataset for Urban Foundation Models* (2025) — arXiv:2511.06337
- MonoScene / VoxFormer / SurroundOcc（3D occupancy）
- MineDojo / Voyager / MineRL（具身 Minecraft）
- 原田实验室主页：mi.t.u-tokyo.ac.jp（Publications / 3D Reconstruction）
