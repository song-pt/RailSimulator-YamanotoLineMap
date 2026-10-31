# 山手线地图重建：进度与断点续作说明

更新：2026-10-30。分支：`generator-test`（未合入 main）。

## 1. 已做的决定

- **不在 Codex 版本上修改，改为从零程序化重建。**
  - 现有 `.railmap` 只有编译好的 iOS AssetBundle：201 个 500m 分区，每个约 66 个按类别合并的网格，每区约 74 万三角面。没有源码，也没有 LOD 和碰撞体。
  - 现有地图的问题出在生成逻辑上：
    - 线路高程是平滑后的地形，没有和地面统一基准（东京站 -16.4m、品川 -21.6m），所以会出现"地下段"、能从地下看见城市。
    - 接触网柱等距排布，不避让站房和站台。
- **测试段**：东京→神田→秋叶原（山手线内回り，主线里程 s≈650–3250m，约 2.6km）。
- **测试交付**：用 Blender 建模，渲染一段沿轨道前进的司机视角 MP4。先不打 Unity 包。

## 2. 技术链路

```
OSM（轨道/站台/道路/水面） ┐
国土地理院 DEM5A（地面）    ├─> network.py   统一高程基准 + 结构类型
PLATEAU LOD2（真实建筑）    ┤   model.py     部件化生成 + 规则避让
Poly Haven CC0（PBR 贴图）  ┘   validate.py  限界/穿模校验（门禁）
        -> scene.pkl（按材质的网格）-> Blender 组装 + 渲染 MP4
        -> （之后）glTF -> Unity 编辑器脚本 -> iOS AssetBundle -> .railmap
```

### 关键规则（都在代码里自动执行）

**高程**（`network.py`）
- 主线轨顶 = 平滑后的地面 + 7.6m（高架）。
- 其他线路 = 主线基准 + 按 OSM layer 加偏移：layer2 +8.4m，layer3 +16.8m，layer4 +11.0m。
- 长度不足 80m 的 way 不采信其 layer（这类多是跨路小桥）。
- 在轨道图上解稀疏线性方程做平滑，层间变化形成坡道。实测最大坡度约 21‰。

**结构**（`model.py: build_structures`）
- 每条轨道切成约 10m 一段，每段桥面半宽 2.9m。
  - 下方有更低的轨道：只建桥板加桥墩，挖掉下层轨道 2.25m 范围。
  - OSM 标 bridge=yes（跨路）：钢梁板，下方留空。
  - 其余：实体高架落地。山手线和京浜东北线用砖墙，其他用混凝土。

**站台**
- 由 OSM 的 platform_edge 线和站台多边形推出"轨道 + 里程区间 + 侧"。
- 向外侧打射线找到相邻同层轨道，确定站台宽度。
- 站台边缘统一切到距轨道中心 1.58m，站台面高出轨面 1.1m。
- 附属设施：站台门（山手线和京浜东北线侧）、黄色盲道、雨棚和立柱、站名牌、长椅、楼梯口。

**接触网**
- 沿主线每 45m 做一个横断面，按同层、相邻关系把轨道分组。
- 门型架的柱子放在组外侧 ±2.6m，以及大于 7m 的间隙中间（例如站台中线）。
- 柱子必须同时满足：
  - 距任何轨道 ≥2.35m；
  - 站在桥面或站台上；
  - 不碰站台立柱和其他障碍物。
- 失败时，柱位横向偏移最多 1.2m，横断面纵向挪动最多 ±12m。
- 线高：接触线 5.1m，承力索 6.2m（带弛度），吊弦每 5m 一根。上方有桥时自动压低。

**首都高**
- 都心环状线在测试段内跨越 JR 线，所以单独建模。
- 高度取"地面 +10m"与"下方轨道 +10.2m"中的较大者，坡度限制 5%。
- 桥墩避开轨道。

**建筑**
- 用 PLATEAU LOD2，楼顶和墙面都是真实形状。
- 与轨道走廊（3.6m）、桥面或站台相交的建筑：先裁掉相交部分，再按 LOD1 挤出；剩余太小就删除。
- 东京站丸之内站舍用红砖立面，其余建筑按高度和哈希分配玻璃幕墙、面砖、涂料、商铺等立面。

**校验**（`validate.py`）
- 车辆限界分三层（自轨面算起的高度，及距轨道中心的半宽）：
  - 0–1.15m：半宽 1.50m；
  - 1.15–4.10m：半宽 1.75m；
  - 4.10–5.60m（受电弓区）：半宽 1.00m。
- 所有实体障碍物都与限界求交，并检查行驶线轨面是否高于地面。

## 3. 当前状态（2026-10-31）

- `model.py` 已完整跑通，全流程约 60 秒，约 530 万三角面。
- 新增站台齐平处理：同一站台两侧的轨道强制同一高程，坡道被推到站外。
- 与主线并行的同层轨道跟随主线高程。
- 限界校验：驾驶线 ±120m 范围内剩余 43 处，全部是站台或雨棚的局部高差。
  - 主要集中在秋叶原总武线站台，以及有乐町一侧（相机起点之后，画面里看不到）。
  - 待修。
- 驾驶线轨面离地最低 6.26m，不存在地下段。
- 已出 4 张测试静帧：东京、东京–神田高架、神田、秋叶原。
  - 渲染设置：Cycles，960×540，12 采样，约 30 秒/帧。
- 视频按用户要求暂不渲染：全程约 3200 帧，耗时太长。

## 4. 断点续作步骤

```sh
git clone -b generator-test https://github.com/song-pt/RailSimulator-YamanotoLineMap
cd RailSimulator-YamanotoLineMap
sh generator/setup_env.sh ~/railwork
cd generator
python3 -I -c "import sys,os;sys.path.insert(0,'.');import model,validate;m=model.Model(os.path.expanduser('~/railwork/data')).build();m.log+=validate.check(m);m.export(os.path.expanduser('~/railwork/scene'));print(m.log)"
~/railwork/blender/blender -b --python blender_scene.py -- ~/railwork/scene ~/railwork/data/tex ~/railwork/render --still 1000,1500,2060,2930 --samples 12 --res 960 540
```

下一步待办：
1. 清掉剩余的站台和雨棚高差问题。
2. 改善外观：钢轨颜色；砖拱高架的拱形细节；接触网补柱（每个横断面约 40% 的轨道组未能立柱，见 `rejects` 统计）。
3. 扩展到全线。
4. 导出 glTF，写 Unity 打包脚本和 `build_map.sh`。

## 5. 文件说明

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 新机器一键装环境、下载数据 |
| `fetch_osm.py` | OSM API 分块下载（Overpass 在这个环境里不稳定，没用） |
| `dem.py` | 国土地理院 DEM5A 瓦片下载和双线性采样 |
| `fetch_plateau.py` | 用 HTTP Range 从 2GB 的 PLATEAU zip 里只取需要的网格文件 |
| `plateau.py` | CityGML → `buildings.pkl` |
| `fetch_textures.py` | Poly Haven 贴图（需要带 User-Agent） |
| `geo.py` | 投影：以东京站为原点，x 向东、y 向北，单位米；OSM 读取工具 |
| `network.py` | 轨道网和高程求解 |
| `meshkit.py` | 向量化网格工具：扫掠、批量盒子、圆柱、导线、多边形挤出 |
| `model.py` | 场景生成主程序 |
| `validate.py` | 限界校验 |

数据来源和许可：OpenStreetMap（ODbL）、国土地理院、PLATEAU（国土交通省，CC BY 4.0 兼容）、Poly Haven（CC0）、Noto Sans SC（OFL）。发布时需要署名。
