# vtk2Cesium

`vtk2Cesium` 正在实现 VTK 体素数据到 CesiumJS 实验性体素瓦片管线的转换。

阶段 0–4 已完成：CesiumJS 1.143 兼容性探针、VTI 读取、WGS84/ENU 定位、GLB 体素写入，以及 SDK/YAML/CLI 转换管线。探针用于验证以下实验格式：

- 3D Tiles 1.1 `tileset.json`
- `3DTILES_content_voxels`
- 隐式八叉树与 subtree availability
- glTF 2.0 `EXT_primitive_voxels`
- glTF `EXT_structural_metadata` property attributes

此体素 API 和相关扩展仍是实验性的，可能不遵循 CesiumJS 的常规弃用策略。

## 阶段 0 快速验证

```bash
python -m vtk2cesium.probe outputs/probe
python -m vtk2cesium.validate outputs/probe
python -m http.server 8000 --directory .
```

然后访问 `http://localhost:8000/examples/viewer/`。页面使用固定的 CesiumJS 1.143 CDN 版本，并通过 `Cesium3DTilesVoxelProvider` + `VoxelPrimitive` 加载 `outputs/probe/tileset.json`。

查看页支持以下 URL 参数（`?` 后追加）：

- `tileset=路径`：指定要加载的 tileset（默认 `../../outputs/geology-shandong-stage5/tileset.json`）。
- `georeferenced=0|1`：强制覆盖"是否自带地理 transform"的自动判断。
- `underground=0|1`：覆盖地下查看模式；默认**自动**——当包围盒延伸到地表以下时自动开启（关闭相机碰撞下潜限制、地壳半透明、关闭地形深度遮挡）。
- `globe=0`：进一步隐藏地壳（适用于只想看纯地下体素）。

地下地质体（如 `outputs/geology-shandong-stage5`，−20 km~+0.3 km）默认即可下潜观察；风场等地表以上数据不会触发地下模式。

## VTI 读取 API

```python
from vtk2cesium.readers import inspect_vti, read_vti

inspection = inspect_vti("model.vti")
for field in inspection.fields:
    print(field.association, field.name, field.components)

dataset = read_vti("model.vti", field_name="temperature", association="point")
values = dataset.field("temperature").values  # shape: (z, y, x[, components])
```

- Point Data 使用 VTI 点维度 `(nx, ny, nz)`。
- Cell Data 使用单元维度 `(nx-1, ny-1, nz-1)`。
- NumPy 统一存储为 `(z, y, x[, components])`，X 在二进制 buffer 中移动最快。
- 当字段选择存在歧义时，必须显式指定字段名和/或关联位置。

## 地理定位与标量预处理

地理定位必须显式给出 WGS84 经度、纬度和椭球高。局部 VTK 坐标约定为 ENU：`x=East`、`y=North`、`z=Up`。

```python
from vtk2cesium.geo import GeoReference, local_box_from_bounds

reference = GeoReference(longitude=116.3913, latitude=39.9075, height=1200.0)
transform = reference.tileset_transform()  # 3D Tiles column-major 16 元素
box = local_box_from_bounds(dataset.bounds)
```

标量预处理默认保留有限原始值，并为 NaN/Inf 和阈值外数据生成同形状 mask；也可选择线性、对数或分段映射。

```python
from vtk2cesium.transfer import (
    ScalarMapping,
    ScalarPreprocessConfig,
    TransferFunction,
    TransferPoint,
    preprocess_scalar,
)

result = preprocess_scalar(
    values,
    ScalarPreprocessConfig(
        mapping=ScalarMapping.LINEAR,
        source_range=(-4.0, 6.6),
        threshold=(-2.0, None),
    ),
)

transfer = TransferFunction((
    TransferPoint(0.0, (0.02, 0.34, 0.72), 0.15),
    TransferPoint(1.0, (0.96, 0.32, 0.12), 0.82),
))
rgba_lut = transfer.lookup_table(256)
```

## 阶段 3：GLB 体素瓦片写入

```python
from vtk2cesium.geo import GeoReference
from vtk2cesium.readers import read_vti
from vtk2cesium.transfer import ScalarMapping, ScalarPreprocessConfig
from vtk2cesium.writer import write_voxel_tileset

source = read_vti("samples/at.vti")
tileset = write_voxel_tileset(
    source,
    "outputs/at-stage3",
    field_name="density",
    georeference=GeoReference(116.3913, 39.9075, 1200.0),
    preprocess=ScalarPreprocessConfig(mapping=ScalarMapping.LINEAR),
)
```

输出结构：

```text
outputs/at-stage3/
├── tileset.json
├── subtrees/0.0.0.0.subtree
└── content/0.0.0.0.glb
```

写入器先在同级临时目录生成完整数据集，再原子替换目标；默认拒绝覆盖非空目录。阶段 3 目前只编码单分量 FLOAT32 标量，独立 validity mask 留给后续格式扩展。

## 阶段 4：SDK、YAML 与 CLI

SDK：

```python
from vtk2cesium import ConvertConfig, convert_vti

config = ConvertConfig(
    input="samples/at.vti",
    output="outputs/at-sdk",
    field_name="density",
    association="point",
    georeference={
        "longitude": 116.3913,
        "latitude": 39.9075,
        "height": 1200.0,
    },
    preprocess={"mapping": "linear"},
)
result = convert_vti(config)
print(result.tileset)
```

YAML：

```yaml
input: samples/at.vti
output: outputs/at-yaml
field_name: density
association: point
georeference:
  longitude: 116.3913
  latitude: 39.9075
  height: 1200.0
preprocess:
  mapping: linear
overwrite: false
```

PowerShell CLI：

```powershell
vtk2cesium inspect "samples\at.vti"

vtk2cesium convert `
  "samples\at.vti" `
  "outputs\at-cli" `
  --field density `
  --association point `
  --lon 116.3913 `
  --lat 39.9075 `
  --height 1200 `
  --mapping linear

vtk2cesium convert --config "convert.yaml"
vtk2cesium validate "outputs\at-cli"
```

CLI 参数覆盖 YAML 配置；`--json` 可输出机器可读结果。退出码：`2` 配置/用法，`3` 输入或字段，`4` 输出冲突，`5` 产物验证失败。

## 阶段 5：结构化分块与多层 LOD

阶段 5 在阶段 4 的单根瓦片之上增加了隐式八叉树多层结构：固定尺寸瓦片（带边界填充）、`2×2×2` 掩码感知 LOD 聚合，以及通用 OCTREE 隐式子树 availability bitstream。

提供 `--available-levels`（≥2，由数据维度推导统一 tile 尺寸）或 `--tile-dimensions`（逗号分隔 `x,y,z`，推导最小层级数）其中之一即可开启多层；两者都省略时回落到阶段 4 的单个根瓦片。

CesiumJS 要求所有隐式瓦片使用**统一** `tile dimensions`，因此数据会被 pad 到 `capacity = tile_dim × 2^(levels-1)`，再按层级写出。粗层（root）是该体积的降采样概览，细层补全全分辨率。

SDK：

```python
from vtk2cesium import ConvertConfig, convert_vti
from vtk2cesium.config import TilingConfig

config = ConvertConfig(
    input="samples/at.vti",
    output="outputs/at-stage5",
    field_name="density",
    association="point",
    georeference={
        "longitude": 116.3913,
        "latitude": 39.9075,
        "height": 1200.0,
    },
    preprocess={"mapping": "linear", "source_range": (-3.988, 6.530)},
    tiling=TilingConfig(available_levels=3),
)
result = convert_vti(config)
print(result.tileset, result.value_count)
```

YAML：

```yaml
input: samples/at.vti
output: outputs/at-stage5
field_name: density
association: point
georeference:
  longitude: 116.3913
  latitude: 39.9075
  height: 1200.0
preprocess:
  mapping: linear
  source_range: [-3.988, 6.530]
tiling:
  available_levels: 3
```

PowerShell CLI：

```powershell
vtk2cesium convert `
  "samples\at.vti" `
  "outputs\at-stage5" `
  --field density `
  --association point `
  --lon 116.3913 `
  --lat 39.9075 `
  --height 1200 `
  --mapping linear `
  --source-min -3.988 `
  --source-max 6.530 `
  --available-levels 3

# 或显式指定 tile 尺寸（自动推导层级）：
#   --tile-dimensions 45,24,12

vtk2cesium validate "outputs\at-stage5"
```

多层产物结构（`available_levels=3` 时共 73 个内容瓦片：1 个根 + 8 + 64）：

```text
outputs/at-stage5/
├── tileset.json
├── subtrees/0.0.0.0.subtree
├── subtrees/0.0.0.0.bin
└── content/{level}.{x}.{y}.{z}.glb   # 共 73 个
```

查看页（`examples/viewer/`）会自动检测 tileset 是否自带非单位 `transform`：地理定位产物（含 `transform`）直接用其坐标，不再叠加局部 `modelMatrix`；局部探针（无 `transform`）仍用放大定位矩阵。因此 `outputs/at-stage5` 与 `outputs/at-stage4-cli` 均能正确落在北京上空，而不会被甩到太空。`--height 1200` 是预期高度（北京上空约 1.2 km）；如需贴地判断位置可用 `--height 0` 重新 convert。

对于延伸到地表以下的产物（如 `outputs/geology-shandong-stage5`，局部 z 从 −20 km 到约 0），查看页会**自动开启地下模式**：关闭相机碰撞下潜限制（`enableCollisionDetection=false`）、把地壳设为半透明（`globe.translucency`）、关闭地形深度遮挡（`depthTestAgainstTerrain=false`），于是地下体素不再被地球挡住。也可用 `?globe=0` 直接隐藏地壳、`?underground=0` 强制关闭。

## 大范围测试数据

`samples/generate_large_vti.py` 用 numpy + VTK 生成二进制 `.vti`，用于压测读取→预处理→地理定位→多层分块/LOD→GLB→离线校验全链路（非单位 spacing 也会触发地理尺度正确性校验）。

```powershell
# 济南历下区风场（默认 200×200×40，约 160 万点；字段 u/v/w/speed/temperature）
python samples/generate_large_vti.py wind --output "samples\wind_lixia.vti"

# 山东较大范围地质体（默认 256×256×64，约 419 万点；字段 density/porosity，地表向下 20 km）
python samples/generate_large_vti.py geo  --output "samples\geology_shandong.vti"

# 生成后直接转多层并校验（风场 3 层 / 地质体 4 层）
vtk2cesium convert "samples\wind_lixia.vti"       "outputs\wind-lixia-stage5"       --field speed  --lon 117.07 --lat 36.66 --height 0 --mapping linear --available-levels 3
vtk2cesium convert "samples\geology_shandong.vti" "outputs\geology-shandong-stage5" --field density --lon 117.00 --lat 36.60 --height 0 --mapping linear --available-levels 4
```

- 风场产物 73 个瓦片，根盒约 **15 km × 15 km × 2 km**；地质体产物 585 个瓦片，根盒约 **100 km × 100 km × 20 km**。
- 多层根包围盒按米制计算（`origin + capacity × spacing`），与无尺度的 ENU→ECEF `transform` 一致；`--tile-dimensions` 亦可用。

## 阶段 6：矢量叠加（箭头 / 流线）

第二档风场可视化：**不改造单标量生产管线**，而是从原始 VTI 的 `u/v/w` 速度分量生成一份**独立的 CZML 叠加图层**（箭头 glyph + RK4 流线），用与体素瓦片**同一套** ENU→ECEF `transform` 地理定位，因此二者在查看页精确对齐、可独立开关。

```powershell
# 生成独立于 convert 的矢量叠加（箭头 + 流线），写到 outputs\wind-lixia-vectors\
vtk2cesium vector "samples\wind_lixia.vti" "outputs\wind-lixia-vectors" `
  --lon 117.07 --lat 36.66 --height 0 `
  --step 16 --arrow-length 500 --streamlines 120 --streamline-steps 60 --streamline-step 220
```

- `--step`（默认 16，或 `x,y,z`）控制箭头降采样步长；`--arrow-length` 为箭头长度（米）；`--streamlines`/`--streamline-steps`/`--streamline-step` 控制流线种子数、步数、步长。
- 输出按 ~48 KB 分片的 `arrows-<i>.czml` / `streamlines-<i>.czml` 加一份 `vectors-manifest.json` 索引；位置均为 ECEF cartesian、按风速着色（蓝→红），`arcType: NONE` 保证是 3D 直线段。分片是为了绕开某些静态服务器/预览代理对响应体的 64 KiB 上限（大文件会被截断成残缺 JSON），全部箭头与流线无损保留。
- 单标量 `convert` 路径完全不变；`vector` 是独立的 CLI 子命令，复用 `geo.GeoReference` 但不触碰体素 GLB/tileset 写入器。

查看页（`examples/viewer/`）的叠加层目录写死为默认值 `../../outputs/wind-lixia-vectors`（`viewer.js` 顶部的 `DEFAULT_VECTORS_BASE`），即 `vector` 命令的默认输出位置：

- `?vectors=<目录>` 换成别的矢量产物目录；
- `?vectors=0` 或 `?vectors=off` 关闭叠加层。

加载 `vectors-manifest.json` 列出的全部分片（旧的单文件 `arrows.czml` / `streamlines.czml` 仍作兜底），面板出现「箭头 glyph / 流线」两个开关——每组一个开关，分片对用户不可见。详情面板的 `Vectors base` 显示实际使用的目录；加载失败时直接列出确切的 URL 与 HTTP 状态码，便于定位路径问题。

注意矢量目录必须与体素 tileset 用**同一套** `--lon/--lat/--height`，否则两者不重合。地质体样本只有 `density`/`porosity`、没有 `u/v/w`，因此没有对应的矢量叠加，用它做底图时需要 `?vectors=off`。验证：箭头基点 ECEF 与体素 transform 误差 **0.0 m**，全部落进球素根盒内。

## 兼容约定

- `tileset.json` 中 dimensions 为 `[x, y, z]`。
- BOX 体素 glTF 的 `EXT_primitive_voxels.dimensions` 按 CesiumJS 官方样例写为 `[x, z, y]`，用于适配 glTF Y-up。
- 标量 buffer 按 X 最快、然后 Y、最后 Z 的顺序展平。
- 首个探针只有根瓦片可用，subtree 使用 JSON constant availability。
