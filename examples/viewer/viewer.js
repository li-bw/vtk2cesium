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

// 矢量叠加层默认目录。换数据集时用 ?vectors=<目录> 覆盖，?vectors=0|off 关闭。
// 无显式覆盖时，从 tileset 路径推导矢量目录（位置无关，不假设 doc-root 是项目根）。
const DEFAULT_VECTORS_BASE = "../../outputs/wind-lixia-vectors";

function resolveVectorsBase(override, tilesetPath) {
  // ?vectors=0|off 彻底关闭叠加层；显式 ?vectors=<dir> 直接采用（相对/绝对均可）
  if (override === "0" || override === "off") return null;
  if (override) return new URL(override, window.location.href).href.replace(/\/+$/, "");
  // 从 tileset 路径推导：约定矢量目录与 tileset 同父目录，名为 <name>-vectors
  // （如 wind-lixia-stage5/tileset.json -> wind-lixia-vectors）。 geology 等不再错指 wind-lixia。
  if (tilesetPath) {
    const tileDir = new URL(tilesetPath, window.location.href).href.replace(/\/[^/]*$/, "");
    const name = tileDir.replace(/.*\//, "").replace(/-stage\d+$/, "");
    const parent = tileDir.replace(/\/[^/]*$/, "");
    return `${parent}/${name}-vectors`;
  }
  return DEFAULT_VECTORS_BASE;
}

async function loadVectorOverlay(viewer, base) {
  if (!base) return null;

  const controlRoot = document.getElementById("vector-controls");
  const makeToggle = (labelText, dataSources) => {
    const wrapper = document.createElement("label");
    wrapper.className = "toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      for (const ds of dataSources) ds.show = checkbox.checked;
    });
    const text = document.createElement("span");
    text.textContent = labelText;
    wrapper.append(checkbox, text);
    controlRoot.append(wrapper);
  };

  // 唯一的加载入口：vectors-manifest.json 索引 arrows-<i>.czml / streamlines-<i>.czml
  // 分片（单文件保持在 64 KiB 传输上限以下）。每个分片自带 document 包，是独立的 CZML 流。
  const loadShard = (fileName) => Cesium.CzmlDataSource.load(`${base}/${fileName}`);

  let arrowSources = [];
  let lineSources = [];
  let error = "";
  try {
    const manifestUrl = `${base}/vectors-manifest.json`;
    const response = await fetch(manifestUrl);
    if (!response.ok) throw new Error(`${manifestUrl} -> HTTP ${response.status}`);
    const manifest = await response.json();
    for (const name of manifest.arrows || []) arrowSources.push(await loadShard(name));
    for (const name of manifest.streamlines || []) lineSources.push(await loadShard(name));
    if (!arrowSources.length && !lineSources.length) throw new Error("manifest 未列出任何分片");
  } catch (loadError) {
    arrowSources = [];
    lineSources = [];
    error = `${(loadError && loadError.message) || loadError}`;
  }

  const addGroup = (label, sources) => {
    let count = 0;
    for (const ds of sources) {
      viewer.dataSources.add(ds);
      ds.show = true;
      count += ds.entities.values.length;
    }
    // 每组一个开关，而不是每个分片一个：分片只是传输细节。
    if (sources.length) makeToggle(label, sources);
    return count;
  };

  const arrowCount = addGroup("箭头 glyph", arrowSources);
  const lineCount = addGroup("流线", lineSources);

  if (arrowCount > 0 || lineCount > 0) {
    controlRoot.hidden = false;
    return { arrows: arrowCount, streamlines: lineCount, source: base };
  }
  return { arrows: 0, streamlines: 0, source: null, error: error || "未找到矢量产物", base };
}

async function loadParticleField(viewer, vectorsBase, particlesOverride) {
  // 第三档：解耦的粒子风场。复用矢量叠加的 base 目录，或 ?particles=<目录> 覆盖；
  // ?particles=0|off 关闭。只实例化独立的 WindParticles 类，不触碰体素/箭头/流线。
  if (particlesOverride === "0" || particlesOverride === "off") return null;
  const base = particlesOverride
    ? new URL(particlesOverride, window.location.href).href.replace(/\/+$/, "")
    : vectorsBase;
  if (!base) return null;
  const manifestUrl = `${base}/velocity-field-manifest.json`;
  const probe = await fetch(manifestUrl);
  if (!probe.ok) return null; // 没有速度场产物就不显示开关

  const wind = new WindParticles(viewer, { fieldUrl: manifestUrl });
  try {
    await wind.init();
  } catch (error) {
    console.warn("粒子风场初始化失败：", error);
    return { error: (error && error.message) || String(error) };
  }
  wind.start();

  const controlRoot = document.getElementById("vector-controls");
  const wrapper = document.createElement("label");
  wrapper.className = "toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = true;
  checkbox.addEventListener("change", () => wind.setVisible(checkbox.checked));
  const text = document.createElement("span");
  text.textContent = "粒子风场";
  wrapper.append(checkbox, text);
  controlRoot.append(wrapper);
  controlRoot.hidden = false;
  return { particles: wind.opts.particleCount, source: base };
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
    //geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    //timeline: false,
    lockButton: true,
    animation: true,
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


  // 城市白膜（可选叠加，默认不加载）：避免内网私有地址离线报错、也不劫持相机。
  // 用法：URL 加 ?city=<tileset.json 完整地址>，例如
  // ?city=http://192.168.1.5:8888/.../JiNanBuilding/Tile/tileset.json
  const cityUrl = new URLSearchParams(window.location.search).get("city");
  if (cityUrl && cityUrl !== "0" && cityUrl !== "off") {
    Cesium.Cesium3DTileset.fromUrl(cityUrl)
      .then((set) => { viewer.scene.primitives.add(set); })
      .catch((err) => { console.warn(`城市白膜加载失败（已跳过）：${cityUrl}`, err); });
  }



  const parameters = new URLSearchParams(window.location.search);
  const tilesetPath = parameters.get("tileset") ?? "../../outputs/wind-lixia-stage5/tileset.json";
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
  // 体素着色由 makeVoxelShader 生成：颜色端点 / 透明度 / 按值透明 通过 uniforms
  // 暴露给面板，运行时改 uniforms 即实时生效；property 字段切换会重建 shader。
  const shader = makeVoxelShader(propertyName, {
    colorLow: Cesium.Color.fromCssColorString("#0557b8"),
    colorHigh: Cesium.Color.fromCssColorString("#f5511f"),
    alpha: 0.6,
    byValue: false,
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
  buildVoxelStyleControls(viewer, primitive, provider);

  // 矢量叠加层（第二档）：从同一套 ENU->ECEF transform 派生的 arrows/streamlines CZML。
  // 与体素瓦片完全解耦——只做加法图层，不改动 VoxelPrimitive 加载逻辑。
  const vectorsBase = resolveVectorsBase(parameters.get("vectors"), tilesetPath);
  const overlay = await loadVectorOverlay(viewer, vectorsBase);
  const particles = await loadParticleField(viewer, vectorsBase, parameters.get("particles"));

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
    ["Vector overlay", overlay && overlay.source
      ? `${overlay.arrows} arrows / ${overlay.streamlines} lines`
      : "none"],
    ["Vectors base", (overlay && (overlay.source || overlay.base)) || "—"],
    // 失败时把确切的 URL 与状态码写进面板，路径不对可以直接在页面上看出来。
    ...(overlay && !overlay.source && overlay.error
      ? [["矢量图层加载失败", overlay.error]]
      : []),
    ["Particle field", particles && particles.source
      ? `${particles.particles} particles`
      : (particles && particles.error ? "error" : "none")],
    ...(particles && particles.error
      ? [["粒子风场加载失败", particles.error]]
      : []),
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

// ---- 体素样式面板：属性字段 / 颜色 / 透明度，实时生效 ----
// 颜色端点、整体透明度、按值透明通过 CustomShader uniforms 实时修改；
// 切换 property 字段必须重建 shader（属性名编译进 GLSL），赋值 primitive.customShader 即时生效。
function makeVoxelShader(propertyName, opts) {
  const low = opts.colorLow;
  const high = opts.colorHigh;
  return new Cesium.CustomShader({
    uniforms: {
      u_colorLow: {
        type: Cesium.UniformType.VEC3,
        value: new Cesium.Cartesian3(low.red, low.green, low.blue),
      },
      u_colorHigh: {
        type: Cesium.UniformType.VEC3,
        value: new Cesium.Cartesian3(high.red, high.green, high.blue),
      },
      u_alpha: { type: Cesium.UniformType.FLOAT, value: opts.alpha },
      u_byValue: { type: Cesium.UniformType.FLOAT, value: opts.byValue ? 1.0 : 0.0 },
    },
    fragmentShaderText: `
      void fragmentMain(FragmentInput fsInput, inout czm_modelMaterial material) {
        float value = fsInput.metadata.${propertyName};
        vec3 c = mix(u_colorLow, u_colorHigh, clamp(value, 0.0, 1.0));
        material.diffuse = c;
        float a = (u_byValue > 0.5)
          ? smoothstep(0.08, 0.32, value) * u_alpha
          : u_alpha;
        material.alpha = a;
      }
    `,
  });
}

function buildVoxelStyleControls(viewer, primitive, provider) {
  const root = document.getElementById("voxel-controls");
  if (!root) return;
  const names = (provider.names && provider.names.length) ? provider.names : ["density"];
  const state = {
    propertyName: names[0],
    colorLow: "#0557b8",
    colorHigh: "#f5511f",
    alpha: 0.6,
    byValue: false,
  };

  const rebuildShader = () => {
    primitive.customShader = makeVoxelShader(state.propertyName, {
      colorLow: Cesium.Color.fromCssColorString(state.colorLow),
      colorHigh: Cesium.Color.fromCssColorString(state.colorHigh),
      alpha: state.alpha,
      byValue: state.byValue,
    });
  };

  const setVec3Uniform = (name, hex) => {
    const s = primitive.customShader;
    if (!s || !s.uniforms[name]) return;
    const c = Cesium.Color.fromCssColorString(hex);
    s.uniforms[name].value = new Cesium.Cartesian3(c.red, c.green, c.blue);
  };
  const setFloatUniform = (name, value) => {
    const s = primitive.customShader;
    if (s && s.uniforms[name]) s.uniforms[name].value = value;
  };

  root.innerHTML = "";

  // 属性字段选择（切换会重建 shader）
  const propField = document.createElement("div");
  propField.className = "field";
  const propLabel = document.createElement("label");
  propLabel.textContent = "属性字段";
  const propSelect = document.createElement("select");
  for (const n of names) {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    propSelect.append(opt);
  }
  propSelect.value = state.propertyName;
  propSelect.addEventListener("change", () => {
    state.propertyName = propSelect.value;
    rebuildShader();
  });
  propField.append(propLabel, propSelect);

  // 低值颜色
  const lowField = document.createElement("div");
  lowField.className = "field";
  const lowLabel = document.createElement("label");
  lowLabel.textContent = "低值颜色";
  const lowInput = document.createElement("input");
  lowInput.type = "color";
  lowInput.value = state.colorLow;
  lowInput.addEventListener("input", () => {
    state.colorLow = lowInput.value;
    setVec3Uniform("u_colorLow", state.colorLow);
  });
  lowField.append(lowLabel, lowInput);

  // 高值颜色
  const highField = document.createElement("div");
  highField.className = "field";
  const highLabel = document.createElement("label");
  highLabel.textContent = "高值颜色";
  const highInput = document.createElement("input");
  highInput.type = "color";
  highInput.value = state.colorHigh;
  highInput.addEventListener("input", () => {
    state.colorHigh = highInput.value;
    setVec3Uniform("u_colorHigh", state.colorHigh);
  });
  highField.append(highLabel, highInput);

  // 透明度滑块（实时改 uniform）
  const alphaField = document.createElement("div");
  alphaField.className = "field";
  const alphaLabel = document.createElement("label");
  alphaLabel.textContent = "透明度";
  const alphaWrap = document.createElement("div");
  alphaWrap.style.display = "flex";
  alphaWrap.style.alignItems = "center";
  alphaWrap.style.gap = "8px";
  const alphaInput = document.createElement("input");
  alphaInput.type = "range";
  alphaInput.min = "0";
  alphaInput.max = "100";
  alphaInput.value = String(Math.round(state.alpha * 100));
  const alphaValue = document.createElement("span");
  alphaValue.className = "value";
  alphaValue.textContent = state.alpha.toFixed(2);
  alphaInput.addEventListener("input", () => {
    state.alpha = Number(alphaInput.value) / 100;
    alphaValue.textContent = state.alpha.toFixed(2);
    setFloatUniform("u_alpha", state.alpha);
  });
  alphaWrap.append(alphaInput, alphaValue);
  alphaField.append(alphaLabel, alphaWrap);

  // 按值透明（低值隐去，实时改 uniform）
  const byValueField = document.createElement("div");
  byValueField.className = "field checkbox";
  const byValueInput = document.createElement("input");
  byValueInput.type = "checkbox";
  byValueInput.checked = state.byValue;
  const byValueLabel = document.createElement("label");
  byValueLabel.textContent = "按值透明（低值隐去）";
  byValueInput.addEventListener("change", () => {
    state.byValue = byValueInput.checked;
    setFloatUniform("u_byValue", state.byValue ? 1.0 : 0.0);
  });
  byValueField.append(byValueInput, byValueLabel);

  root.append(propField, lowField, highField, alphaField, byValueField);
}
