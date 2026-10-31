# 山手线地图重建：交接文档（给 Codex）

分支：`generator-test`（未合入 main）。日期：2026-10-31。作者：Claude（Claude Code 云端会话）。

本文档的目标是让接手者在 **macOS + Unity 6000.6.4f1** 上，把本分支的程序化生成器产出的全线模型编译成游戏可导入的 `.railmap` 地图包。本文分三部分：

- **已经完成的**：生成器，可以产出经过校验的全线三维模型数据；
- **还没做的**：Unity 导入和打包、线路 JSON、`.railmap` 打包。这部分需要接手者完成，第 5 节给出格式要求和建议实现；
- **已知问题**：见第 7 节。

效果参考图在 `docs/renders/`：

| 文件 | 内容 |
|---|---|
| `01_…` – `10_…` | 全线第一版的 10 个关键位置 |
| `1_…` – `7_…` | 东京–秋叶原测试段 |

---

## 1. 为什么重建，以及重建的原则

旧地图（Release `ios-full-v1`）是编译好的 AssetBundle，没有源码；高程没有统一基准，轨道出现地下段，接触网柱穿过站房。因此从零改为**程序化生成 + 自动校验**，原则如下，接手后请保持：

1. **所有高度用同一基准**：东京湾平均海面（T.P.），和国土地理院 DEM、PLATEAU 一致。
2. **不允许不封顶的地下段**：
   - 轨道低于街道时，一律是**顶部敞开的掘割**，两侧挡土墙砌到街道高度；
   - 只有 OSM 标了 `tunnel` 的段落才是覆盖段，加顶板。
3. **限界零侵入**：所有实体（柱、墩、站台、雨棚、站台门、建筑、桥板、梁、跨线桥）都登记成 2.5D 障碍物，由 `validate.py` 与每条轨道的车辆限界求交。半宽按自轨面算起的高度分三层：

   | 自轨面高度 | 半宽 |
   |---|---|
   | 0–1.15m | 1.50m |
   | 1.15–4.10m | 1.75m |
   | 4.10–5.60m | 1.00m |

4. **站台**：
   - 边缘距轨道中心 1.58m，站台面高出相邻轨面 1.10m；
   - 同一站台两侧的轨道强制同高；
   - 站台面高度取相邻最低的同层轨面。
5. **接触网**：柱子只能立在桥面或站台上，离任何轨道 ≥2.35m，不进建筑；上方有上层桥时，门型架自动降低，或改为从桥底吊挂。
6. **建筑**：
   - 用 PLATEAU 的 LOD2 模型；
   - 与铁路范围相交超过 25% 的整栋删除，其余部分裁剪；
   - 25m 以下的建筑用哑光的日本住宅风格，每栋一个颜色，窗户按整窗居中排布，顶层留出女儿墙。

---

## 2. 代码结构（`generator/`）

| 文件 | 作用 |
|---|---|
| `fetch_full.py` | 下载全线数据：OSM API 分块、GSI DEM5A、PLATEAU CityGML（HTTP Range 只取需要的网格）。写入 `<work>/data/` |
| `fetch_textures.py` | Poly Haven CC0 贴图（需带 User-Agent） |
| `plateau.py` | CityGML → `buildings.pkl`（只取走廊内） |
| `geo.py` | 投影与 OSM 读取。**本地坐标系：原点为东京站（35.6812N, 139.7671E），x 向东、y 向北、z 向上（T.P. 米）** |
| `dem.py` | DEM 瓦片拼接与双线性采样 |
| `profile.py` | 全线主线纵断面：高架段 = 街道 + 7.6m；路堤、平地、掘割段贴合激光 DEM；覆盖段在洞口间插值；坡度上限 25‰ |
| `network.py` | 轨道网：各线相对主线所在层（layer）加偏移，在轨道图上解稀疏方程得到所有轨道高度 |
| `model.py` | 场景生成主程序：轨道、结构、站台、站台设施、边界墙和挡土墙、砖拱、跨线道路桥、首都高、接触网、地面、建筑，以及导出 |
| `validate.py` | 限界校验和掘割/覆盖统计 |
| `build_full.py` | 全线一键构建（带断点） |
| `blender_scene.py` | 可选：在 Blender 中组装并出静帧，也是**材质参考实现**（见 5.4） |
| `profile_report.py` | 全线纵断面分段报告 |
| `PROGRESS.md` | 详细进度记录 |
| `setup_env.sh` | Linux 云容器用的环境脚本（apt）。**Mac 上请按第 3 节手动安装** |

主线：OSM relation **1972960**（JR山手線，内回り方向，从品川起算）。外回り是 relation **1972920**。

---

## 3. 在 Mac 上生成全线模型

需要 Python 3.11 以上，内存 16GB 以上，磁盘约 30GB 空余。首次下载约 12GB。

```sh
git clone -b generator-test https://github.com/song-pt/RailSimulator-YamanotoLineMap
cd RailSimulator-YamanotoLineMap/generator
python3 -m pip install numpy shapely mapbox_earcut lxml scipy pillow

W=~/railwork/full
python3 -I fetch_full.py $W                      # OSM 349 块、DEM、12 个区的 PLATEAU；约 30–60 分钟
python3 -I fetch_textures.py $W/data/tex         # 17 套 PBR 贴图

# 走廊范围（主线两侧 420m）+ 解析建筑（约 6 分钟）
python3 -I - $W/data <<'EOF'
import sys, pickle; sys.path.insert(0, '.')
from geo import OSM
from shapely.geometry import LineString
o = OSM(sys.argv[1] + '/osm.json')
main = LineString([o.xy(n) for n, _ in o.route_chain(1972960)])
pickle.dump(main.buffer(420), open(sys.argv[1] + '/zone.pkl', 'wb'))
EOF
python3 -I plateau.py $W/data/plateau $W/data/zone.pkl $W/data/buildings.pkl

# 全线构建 + 校验 + 导出；4 核 Linux 上约 30 分钟
RAIL_DATA=$W/data RAIL_SCENE=~/railwork/fullscene python3 -I build_full.py | tee ~/railwork/build.log
```

预期日志末尾：
- `clearance violations [all]: 14 {'platform': 11, 'slab': 3}`
- `main line below street: 3462 m open cutting, 624 m covered; covered sections roofed: True`

如果不是这个结果，说明代码或数据有变化，需要先查明原因。

**断点**：完成"站台设施"阶段后，会写出 `~/railwork/ckpt_structures.pkl`，约 2.8GB。下次运行会从这里继续。**修改了断点之前阶段的代码，比如 `network.py`、站台、结构，必须先删除这个断点文件。**

可选的 Blender 出图（Blender 4.5.x）：

```sh
blender -b --python blender_scene.py -- ~/railwork/fullscene $W/data/tex /tmp/out \
  --still 23520 --samples 12 --res 960 540 --radius 1300
```

`--still` 后面的数字是主线里程（米），例如新宿 23520、池袋 19120、涩谷 27180、东京 6700。

---

## 4. 导出数据格式（`~/railwork/fullscene/`）

导出按**材质 × 1km 分区**拆成多组文件。每组用 `<material>@<tx>_<ty>` 命名，`tx`、`ty` 分别为 `floor(x/1000)` 和 `floor(y/1000)`。

| 文件 | 内容 |
|---|---|
| `.v.npy` | float32 N×3 顶点，本地坐标（米，z 向上） |
| `.f.npy` | int32 M×3 三角形。**右手系 z-up 下从外侧看为逆时针** |
| `.uv.npy` | float32 N×2，**单位为米**，材质自己按尺寸缩放 |
| `.c.npy` | float32 N×3，**线性** RGB 顶点颜色。建筑每栋一色，砖拱填充也用到；其余为 1 |
| `.g.npy` | float32 N×3 立面数据：(墙宽 W, 墙顶相对建筑底高度 H, 是否立面 flag)，供窗户着色器整窗排布用（见 5.4） |

另外：
- `scene.json` 包含：
  - `materials`：材质名列表；
  - `tiles`：`[文件名, 材质名, 分区中心x, 分区中心y]` 的列表；
  - `camera`：内回り主线轨面点，0.5m 间隔，N×3；
  - `cam_s`：对应的里程；
  - `stations`：`[日文名, 英文名, [x,y]]` 的列表；
  - `log`：构建日志和校验结果。
- `sign_<站名>.png` 是站名牌贴图。

在 `model.py` 的 `Model.export` 里把 `T = 1000.0` 改成 `500.0`，就能直接按游戏的 500m 分区导出（推荐，见 5.2）。

---

## 5. 待完成：编译成游戏地图（`.railmap`）

格式要求全部来自游戏源码 `song-pt/RailSimulator` 的 `Source/UnityProject/Assets/Scripts/Maps.cs`、`MapImportJob.cs`、`Simulation.cs`。下面是要点，**以源码为准**。

### 5.1 包结构

`.railmap` 是普通 zip 文件，**不能改名成 .zip 后再导入**。规则：
- 条目数 ≤ 10000，只允许 `.json`、`.bundle`、`.txt` 三种扩展名；
- 单个文件 ≤ 4GB，展开后总大小 ≤ 32GB，zip 本身 ≤ 16GB。

`manifest.json` 对应 `MapManifest`：

```json
{
  "schema": 1, "id": "YAMANOTE_V2_IOS", "name": "山手线 · iOS 全线 V2", "platform": "iOS",
  "unityVersion": "6000.6.4f1", "sourceFingerprint": "<任意，例如生成器 commit>",
  "shared": "shared.bundle", "sharedCrc": 0, "sharedSha256": "<hex>",
  "routeFile": "route.json", "routeSha256": "<hex>",
  "routes": [ {"id": "<内回り id>", "file": "route_inner.json", "sha256": "<hex>"},
              {"id": "<外回り id>", "file": "route_outer.json", "sha256": "<hex>"} ],
  "chunks": [ {"file": "chunk_0_0.bundle", "x": 0, "z": 0, "crc": 0, "sha256": "<hex>", "landmark": false} ]
}
```

- `platform` 必须是 `"iOS"`；`chunks` 最多 5000 个；`id` 里不能有 `/`、`\`、`..`。
- `routeFile` 就是默认线路，一般和内回り那份是同一个文件，可以参照旧地图的写法。
- `crc` 是 `BuildPipeline` 输出的 bundle CRC；`AssetBundle.LoadFromFileAsync` 会校验它，填 0 表示不校验。
- 所有 sha256 都在导入时校验，必须和文件实际内容一致。
- `trafficFile`（道路车流）可以省略。

### 5.2 分区 bundle（`MapWorld`）

- 每个分区是一个 500m × 500m 的方块，`(x, z)` 为方块西南角的 Unity 坐标。游戏用 `c.x+250`、`c.z+250` 作为方块中心计算距离。手机端只加载约 650m 内最近的 12 个分区。
- 每个 bundle **只能包含一个根 GameObject 资源**。加载后它被放在 `localPosition=(x,0,z)`，所以网格坐标要**相对方块角点**。
- 根节点下只保留 Transform、MeshFilter、MeshRenderer、LODGroup，其他组件会被删除。**不要加碰撞体**，游戏不用。
- 游戏按物体名做远处隐藏（`Maps.cs` `LateUpdate`）：
  - 名字含 `Ballast_Stone` 的，超过 200m 隐藏；
  - 名字含 `Fastening` 的，超过 290m 隐藏；
  - 名字含 `Sleeper` 或 `Rubber_Pad` 的，超过 450m 隐藏。

  建议把本生成器的 `sleeper` 物体命名成 `…_Sleeper`、`fastening` 命名成 `…_Fastening`。**道床 `ballast` 是整条棱柱，不能叫 `Ballast_Stone`**，否则 200m 外道床会消失，露出空洞。
- 材质统一放进 `shared.bundle`，分区 bundle 只引用。游戏启动时会先 `LoadAllAssets<Material>()`。

**坐标换算**（本地 z-up 右手系 → Unity y-up 左手系）：Unity `(X, Y, Z) = (x, z, y)`。因为交换了两个轴，**三角形顶点顺序必须反转**（`f[:, ::-1]`），否则面会朝里。

**性能**：
- 目前全线约 3370 万三角面，平均每个 500m 分区约 29 万；车站和建筑密集的分区可能超过 100 万。旧地图约每区 74 万，供参考。
- 建议：
  - 建筑做 LOD：远处用 LOD1 外盒，或者减面；
  - 离主线 25m 以外的轨道已经没有枕木和扣件；
  - 窗户是着色器画的，不额外增加几何。

### 5.3 线路 JSON（`RouteData`，`Simulation.cs` 第 18–25 行）

```json
{ "id": "...", "name": "内回り · 经神田、上野", "loop": true, "limitKmh": 90, "repeatLength": <全长>,
  "points": [ {"x": X, "y": Y, "z": Z, "s": S}, ... ],
  "stations": [ {"id": "...", "nameZh": "东京", "nameJa": "東京", "doorSide": "left"|"right",
                 "stopS": ..., "platformStartS": ..., "platformEndS": ...}, ... ] }
```

`Validate()` 的硬性要求：
- **轨道点**：
  - `points` 数量在 2 到 2,000,000 之间；
  - 坐标绝对值不超过 1e6，`y` 是高度；
  - 里程 `s[0]=0`，严格递增；
  - 相邻两点的距离 `d` 满足 0 < d ≤ 1000，且与里程差的偏差不超过 max(0.05, 3%)。**建议按 2m 间距重采样，`s` 取三维累计弧长**。
- **车站**：
  - 2 到 1000 个，`id` 唯一，`stopS` 严格递增；
  - `platformStartS` ≤ `stopS` ≤ `platformEndS`，且 `platformStartS` < `platformEndS`；
  - `doorSide` 只能是 `left` 或 `right`。
- 游戏的运行计划（`RunPlan.Validate`）要求被选中车站的有效站台长度不小于编组长度 + 0.1m，默认编组 220m（E235，11 节）。**所以每个站的 `platformEndS - platformStartS` 必须 ≥ 220.1m。**
- 循环线：`loop=true`、`repeatLength=全长` 时，`Pose(s)` 会按全长取模，可以无限绕圈跑。旧地图没有设置 `repeatLength`，所以跑到线路末端就停。

**两个方向的线路数据从哪里来**：
- **内回り**：主线就是 relation 1972960 的轨道链。用 `model.camera_path()` 得到轨面点（Network 里的 `net.ms` 和 `net.base_h`），或者直接拼接该 relation 各轨道段的 `tr.pts`，后者更准。
- **外回り**：用 `OSM.route_chain(1972920)` 拼出轨道链，高度取 `model.tracks` 中对应轨道段（`tr.wid`）的 `pts`。
- **车站**：用 `model.platform_spans()` 得到的 `(轨道下标, s0, s1, side)`，挑出属于本方向轨道链的那些区间，投影到该方向的里程上，得到站台起止里程。
  - `stopS` 取站台前端附近（例如 `platformEndS - 5m`）；
  - `doorSide` 由 side 和该轨道段在链中的方向决定：side = +1 表示站台在轨道段前进方向的左侧；如果该段在链中是反向使用的，左右要对调；
  - 站名用 OSM 车站节点的 `name`（日文）和 `name:zh`。
  - 每个方向应当有 **30 个站**。新宿、池袋、涩谷等站在 OSM 里有多个 station 节点，请按站名去重。

### 5.4 材质（URP）

`blender_scene.py` 的 `make_material()` 是完整的参考实现，包括材质列表、贴图、缩放、颜色和粗糙度。贴图都来自 Poly Haven（CC0），对应关系见 `fetch_textures.py` 的 `TEX`。要点：
- UV 单位是米，用 Tiling = 1/贴图尺寸。
- 建筑和屋顶材质要**乘顶点颜色**，`.c.npy` 是线性颜色，Unity 的 Mesh.colors 用 Linear 空间。
- 立面窗户是程序化着色器，见 `layout()`、`windows()`、`lowrise()`：
  - 输入是 UV（u 为沿墙距离，v 为距建筑底的高度）和 `.g.npy` 的 (W, H, flag)；
  - 建议把 `.g.npy` 放进 `uv2`（TEXCOORD1），然后用 Shader Graph 重写；
  - 规则是：窗数取整后在墙面居中，两侧至少留 0.4m；只有整层能放下时才开窗，墙顶留 0.5m 女儿墙；
  - 低层住宅：开间 3.6m，层高 2.9m，**金属度为 0**，墙面粗糙度约 0.92。
- 站名牌：`sign_<站名>.png`。

### 5.5 建议的 Unity 打包实现

1. 新建一个 Unity 6000.6.4f1 空工程（URP，iOS 平台），**不要改游戏工程本身**。
2. 写一个 Editor 脚本（C#）：
   - 读取 `.npy`（格式简单：头部 + 小端数组，可以自己解析），或者先用 Python 转成 `.mesh`、`.obj`、glTF；
   - 按 500m 分区建 GameObject，每个材质一个子物体，**超过 65535 个顶点时用 `IndexFormat.UInt32`**；
   - 在 `Assets/Map/` 下建 Prefab，设置 AssetBundle 名，然后调用 `BuildPipeline.BuildAssetBundles(..., BuildTarget.iOS)`。
3. 用 Python 生成 `route_inner.json`、`route_outer.json` 和 `manifest.json`，计算 sha256 和 CRC（CRC 从 `.manifest` 文件或 `BuildPipeline.GetCRCForAssetBundle` 取），最后压成 `.railmap`。
4. 把以上串成 `build_map.sh`，Unity 用 `-batchmode -nographics -quit -executeMethod` 调用。

**先用一个小范围验证**：东京–秋叶原（主线里程约 6000–9500m）打一个小包，导入手机确认坐标、朝向、站台高度、开门侧都对，再打全线。

---

## 6. 其他已验证的事实

- 全线主线长 34,337m（relation 1972960 链），高差范围约 2–45m。
- 共 2273 段轨道、118 个站台、约 2900 个接触网支撑、219 座跨线道路桥、160 个砖拱。
- 保留 21074 栋建筑，删除 446 栋跨轨站房。
- 掘割共 3.46km：巣鴨–大塚、大塚–池袋（向原）、原宿–涩谷、惠比寿–目黑。
- 覆盖段共 0.62km：目白、新宿南口、目黑一带。

---

## 7. 已知问题（第二版待办）

1. **限界侵入还剩 14 处**：五反田附近货物线旁 11 个站台小格，以及新桥、浜松町、新宿北侧 3 处桥板。用 `validate.check(m)` 的 examples 定位。
2. **掘割边缘仍有斜面**：大塚右侧、莺谷左侧。根源是地面网格每格 6m（`build_ground`）太粗。建议改成沿边界墙裁切地面，或者局部用更密的网格。
3. **目黑覆盖段**造型粗糙，顶板和地面交接不好。
4. **接触网**：非山手线轨道（新干线、货物线等）仍有大于 65m 的无支撑段，见日志 `gap report`；山手线本线已检查。
5. **大站**：跨轨站房这一版直接删除了，新宿、涩谷、品川等站的站房和雨棚偏简单。
6. 全线构建约 30 分钟，主要耗时在站台、建筑和接触网。

---

## 8. 数据来源与许可（发布时需要署名）

OpenStreetMap contributors（ODbL）；国土地理院 DEM5A；Project PLATEAU（国土交通省，CC BY 4.0 兼容）；Poly Haven（CC0）；站名字体 Noto Sans SC（OFL，游戏仓库自带的 `UIFont.ttf`）。
