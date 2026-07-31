/* global Cesium */

const statusElement = document.getElementById("status");
const detailsElement = document.getElementById("details");

function setStatus(message, kind) {
  statusElement.textContent = message;
  statusElement.className = `status ${kind}`;
}

function showDetails(entries) {
  detailsElement.replaceChildren();
  for (const [label, value] of entries) {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    detailsElement.append(term, description);
  }
}

async function fetchJson(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn("无法读取 tileset，按局部体素处理：", error);
    return null;
  }
}

// 将 tileset 的局部包围盒最低角点变换到 ECEF，判断是否有部分位于地表以下。
function tilesetExtendsUnderground(tileset) {
  const root = tileset && tileset.root;
  const box = root && root.boundingVolume && root.boundingVolume.box;
  if (!Array.isArray(box) || box.length !== 12) return false;
  const transform =
    Array.isArray(root.transform) && root.transform.length === 16
      ? Cesium.Matrix4.fromArray(root.transform)
      : Cesium.Matrix4.IDENTITY;
  const center = Cesium.Cartesian3.fromArray(box, 0);
  const axes = [
    Cesium.Cartesian3.fromArray(box, 3),
    Cesium.Cartesian3.fromArray(box, 6),
    Cesium.Cartesian3.fromArray(box, 9),
  ];
  const offset = new Cesium.Cartesian3();
  const corner = new Cesium.Cartesian3();
  const ax = new Cesium.Cartesian3();
  const ay = new Cesium.Cartesian3();
  const az = new Cesium.Cartesian3();
  for (let sx = -1; sx <= 1; sx += 2) {
    for (let sy = -1; sy <= 1; sy += 2) {
      for (let sz = -1; sz <= 1; sz += 2) {
        Cesium.Cartesian3.multiplyByScalar(axes[0], sx, ax);
        Cesium.Cartesian3.multiplyByScalar(axes[1], sy, ay);
        Cesium.Cartesian3.multiplyByScalar(axes[2], sz, az);
        Cesium.Cartesian3.add(ax, ay, offset);
        Cesium.Cartesian3.add(offset, az, offset);
        Cesium.Cartesian3.add(center, offset, corner);
        Cesium.Matrix4.multiplyByPoint(transform, corner, corner);
        const carto = Cesium.Cartographic.fromCartesian(corner);
        if (carto && carto.height < 0) return true;
      }
    }
  }
  return false;
}

// 开启地下查看：关闭相机碰撞限制（相机可下潜），把地壳设为半透明并关闭地形深度遮挡，
// 这样地表以下的体素不再被地球挡住。
function applyUndergroundMode(viewer, enabled) {
  viewer.scene.screenSpaceCameraController.enableCollisionDetection = !enabled;
  if (enabled) {
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.globe.translucency.enabled = true;
    viewer.scene.globe.translucency.frontFaceAlpha = 0.12;
    viewer.scene.globe.translucency.backFaceAlpha = 0.0;
  }
}

function hasNonIdentityTransform(transform) {
  if (!Array.isArray(transform) || transform.length !== 16) return false;
  for (let index = 0; index < 16; index += 1) {
    const expected = index % 5 === 0 ? 1 : 0;
    if (Math.abs(transform[index] - expected) > 1e-6) return true;
  }
  return false;
}

async function loadVectorOverlay(viewer, base) {
  if (!base) return null;
  const controlRoot = document.getElementById("vector-controls");
  const makeToggle = (labelText, checked, onChange) => {
    const wrapper = document.createElement("label");
    wrapper.className = "toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = checked;
    checkbox.addEventListener("change", () => onChange(checkbox.checked));
    const text = document.createElement("span");
    text.textContent = labelText;
    wrapper.append(checkbox, text);
    controlRoot.append(wrapper);
  };

  const loadOne = async (suffix, label, defaultOn) => {
    const url = new URL(`${base}/${suffix}`, window.location.href).href;
    try {
      const dataSource = await Cesium.CzmlDataSource.load(url);
      viewer.dataSources.add(dataSource);
      dataSource.show = defaultOn;
      makeToggle(label, defaultOn, (value) => { dataSource.show = value; });
      return dataSource.entities.values.length;
    } catch (error) {
      console.warn(`无法加载矢量图层 ${label}：${url}`, error);
      return 0;
    }
  };

  const arrowCount = await loadOne("arrows.czml", "箭头 glyph", true);
  const lineCount = await loadOne("streamlines.czml", "流线", true);
  if (arrowCount > 0 || lineCount > 0) {
    controlRoot.hidden = false;
    return { arrows: arrowCount, streamlines: lineCount };
  }
  return null;
}

async function start() {
  if (typeof Cesium === "undefined") {
    throw new Error("CesiumJS CDN 未加载，请检查网络连接。");
  }

  const osm = new Cesium.OpenStreetMapImageryProvider({
    url: "https://tile.openstreetmap.org/",
  });
  const viewer = new Cesium.Viewer("cesiumContainer", {
    baseLayer: new Cesium.ImageryLayer(osm),
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    timeline: false,
    animation: false,
    infoBox: false,
    selectionIndicator: false,
    navigationHelpButton: false,
    skyBox: false,
  });
  viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#eef3f8");

  // 允许相机穿透地表、进入地下
//viewer.scene.globe.depthTestAgainstTerrain = false;
// 关闭天空包围盒遮挡地下
//viewer.scene.skyBox.show = false;
// 或者启用地下渲染模式（Cesium 1.98+支持）
//viewer.scene.globe.enableLighting = true;

  const parameters = new URLSearchParams(window.location.search);
  const tilesetPath = parameters.get("tileset") ?? "../../outputs/geology-shandong-stage5/tileset.json";
  const tilesetUrl = new URL(tilesetPath, window.location.href).href;
  const tileset = await fetchJson(tilesetUrl);

  // 地理定位 tileset 的 root 已经带 ENU->ECEF transform；若再叠加一个
  // modelMatrix 会变成双重变换，体素被甩到太空。因此优先从 tileset 自动判断，
  // 显式 ?georeferenced= 参数仍可作为覆盖。
  const georeferenced = parameters.has("georeferenced")
    ? parameters.get("georeferenced") === "1"
    : hasNonIdentityTransform(tileset && tileset.root && tileset.root.transform);

  // 地下模式：当地质体延伸到地表以下时自动开启，也可显式 ?underground=0|1 覆盖；
  // ?globe=0 可进一步隐藏（半透明）地壳，便于直接观察地下体素。
  const extendsUnderground = tilesetExtendsUnderground(tileset);
  const underground = parameters.has("underground")
    ? parameters.get("underground") === "1"
    : extendsUnderground;
  if (underground) {
    applyUndergroundMode(viewer, true);
  }
  if (parameters.get("globe") === "0") {
    viewer.scene.globe.show = false;
  }

  const provider = await Cesium.Cesium3DTilesVoxelProvider.fromUrl(tilesetUrl);
  const propertyName = (provider.names && provider.names.length)
    ? provider.names[0]
    : "density";
  const shader = new Cesium.CustomShader({
    fragmentShaderText: `
      void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
        float value = fsInput.metadata.${propertyName};
        material.diffuse = mix(vec3(0.02, 0.34, 0.72), vec3(0.96, 0.32, 0.12), value);
        material.alpha = smoothstep(0.08, 0.32, value) * 0.82;
      }
    `,
  });
  const primitiveOptions = {
    provider,
    customShader: shader,
    calculateStatistics: true,
  };
  if (!georeferenced) {
    primitiveOptions.modelMatrix = Cesium.Matrix4.multiplyByUniformScale(
      Cesium.Transforms.eastNorthUpToFixedFrame(
        Cesium.Cartesian3.fromDegrees(116.3913, 39.9075, 1200.0),
      ),
      600.0,
      new Cesium.Matrix4(),
    );
  }
  const primitive = viewer.scene.primitives.add(
    new Cesium.VoxelPrimitive(primitiveOptions),
  );
  primitive.nearestSampling = true;
  viewer.camera.flyToBoundingSphere(primitive.boundingSphere, { duration: 0.0 });

  // 矢量叠加层（第二档）：从同一套 ENU->ECEF transform 派生的 arrows/streamlines CZML。
  // 与体素瓦片完全解耦——只做加法图层，不改动 VoxelPrimitive 加载逻辑。
  const vectorsBase = parameters.get("vectors");
  const resolvedVectorsBase = vectorsBase && vectorsBase !== "0" && vectorsBase !== "off"
    ? vectorsBase
    : (tilesetPath.includes("wind-lixia") ? "../../outputs/wind-lixia-vectors" : null);
  const overlay = await loadVectorOverlay(viewer, resolvedVectorsBase);

  const dimensions = `${provider.dimensions.x} × ${provider.dimensions.y} × ${provider.dimensions.z}`;
  showDetails([
    ["Shape", provider.shape],
    ["Dimensions", dimensions],
    ["Property", provider.names.join(", ")],
    ["Component", provider.componentTypes.join(", ")],
    ["Levels", String(provider.availableLevels)],
    ["Tileset", tilesetPath],
    ["Georeferenced", String(georeferenced)],
    ["Underground", String(underground)],
    ["Vector overlay", overlay ? `${overlay.arrows} arrows / ${overlay.streamlines} lines` : "none"],
  ]);
  setStatus("体素 provider 已创建，正在渲染根瓦片。", "ready");

  primitive.initialTilesLoaded.addEventListener(() => {
    setStatus("验收通过：根体素瓦片已加载。", "ready");
  });
}

start().catch((error) => {
  console.error(error);
  setStatus(`加载失败：${error.message}`, "error");
  showDetails([["建议", "先运行 python -m vtk2cesium.probe outputs/probe，再从项目根目录启动 HTTP 服务。"]]);
});
