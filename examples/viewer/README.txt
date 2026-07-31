从项目根目录运行：

1. python -m vtk2cesium.probe outputs/probe
2. python -m http.server 8000 --directory .
3. 浏览器访问 http://localhost:8000/examples/viewer/

必须通过 HTTP 访问，不能直接双击 file:// 页面。CesiumJS 版本固定为 1.143。

阶段 4 的地理定位产物（默认加载）可使用：
http://localhost:8000/examples/viewer/?tileset=../../outputs/at-stage4-cli/tileset.json

查看页会**自动检测 tileset 是否自带 transform**：
- 带 ENU→ECEF transform 的地理定位产物（如 at-stage4-cli），直接用 tileset 自身坐标，不再叠加 modelMatrix；
- 不带 transform 的局部探针（如 outputs/probe），按局部体素处理并叠加放大定位矩阵。

`?georeferenced=0|1` 仅作为覆盖参数。若未带该参数，页面从 tileset.json 自动判断。

重要：不要把地理定位 tileset 与局部 modelMatrix 同时应用，否则会双重变换、体素被甩到太空。

验收页使用无需 Cesium ion token 的 OpenStreetMap 在线底图。不要把个人 token 写入源码或提交到版本库。
