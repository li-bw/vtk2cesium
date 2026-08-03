/* global Cesium */
/*
 * WindParticles — a completely decoupled, browser-side particle wind field.
 *
 * Design contract (why it is "independent"):
 *   - It depends ONLY on a Cesium Viewer and a velocity field. It never reads
 *     the voxel tileset, the VoxelPrimitive, or the arrow/streamline CZML.
 *   - The velocity field is supplied either as a ready object (`options.field`)
 *     or fetched from a manifest URL (`options.fieldUrl`). The field carries its
 *     own ENU->ECEF georeference, so the class needs nothing from the host page.
 *   - All advection math (trilinear sampling, ENU<->ECEF) is self-contained
 *     pure logic — no coupling to the Python pipeline or to viewer internals.
 *
 * The field schema (emitted by `vtk2cesium vector --emit-field`):
 *   {
 *     "nx","ny","nz", "origin":[x,y,z], "spacing":[dx,dy,dz],
 *     "georeference":{"longitude","latitude","height"},
 *     "speed_min","speed_max",
 *     "u":[...], "v":[...], "w":[...]   // length nx*ny*nz, index = i + j*nx + k*nx*ny
 *   }
 * Shards are reassembled by `velocity-field-manifest.json`; the viewer only
 * needs to hand this class the manifest URL.
 */

class WindParticles {
  constructor(viewer, options = {}) {
    this.viewer = viewer;
    this.opts = Object.assign(
      {
        field: null, // assembled field object (alternative to fieldUrl)
        fieldUrl: null, // manifest URL or directory URL
        particleCount: 1200,
        maxAge: 6.0, // seconds before a particle respawns
        speedScale: 80.0, // multiplies physical velocity for a watchable flow
        trailLength: 8, // history points per trail; 0 disables trails
        pointSize: 2.5,
        dropRate: 0.002, // per-frame random respawn probability
        colorBySpeed: true,
        fixedColor: null, // Cesium.Color used when colorBySpeed is false
        maxDt: 0.05, // clamp seconds/frame to avoid jumps on tab refocus
      },
      options,
    );

    this._field = null;
    this._frame = null; // Matrix4: local ENU offset -> ECEF
    this._invFrame = null;
    this._bounds = null; // { min:[x,y,z], max:[x,y,z] } in ENU metres
    this._points = null;
    this._lines = null;
    this._particles = [];
    this._running = false;
    this._visible = true;
    this._lastTime = null;
    this._updateHandler = null;
  }

  /** Load + assemble the field (if needed) and build the Cesium collections. */
  async init() {
    if (this.opts.field) {
      this._setField(this.opts.field);
    } else if (this.opts.fieldUrl) {
      await this._loadField(this.opts.fieldUrl);
    } else {
      throw new Error("WindParticles: provide options.field or options.fieldUrl");
    }
    if (!this._field || !this._field.georeference) {
      throw new Error("WindParticles: field is missing georeference");
    }
    const g = this._field.georeference;
    const origin = Cesium.Cartesian3.fromDegrees(g.longitude, g.latitude, g.height || 0);
    this._frame = Cesium.Transforms.eastNorthUpToFixedFrame(origin);
    this._invFrame = Cesium.Matrix4.inverse(this._frame, new Cesium.Matrix4());

    const f = this._field;
    const max = [
      f.origin[0] + (f.nx - 1) * f.spacing[0],
      f.origin[1] + (f.ny - 1) * f.spacing[1],
      f.origin[2] + (f.nz - 1) * f.spacing[2],
    ];
    this._bounds = { min: f.origin.slice(), max };

    this._points = this.viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
    if (this.opts.trailLength > 0) {
      this._lines = this.viewer.scene.primitives.add(new Cesium.PolylineCollection());
    }

    this._particles = [];
    for (let i = 0; i < this.opts.particleCount; i += 1) {
      const p = { enu: [0, 0, 0], age: 0, history: [], point: null, line: null, speed: 0 };
      this._seed(p);
      const ecef = this._enuToEcef(p.enu);
      const color = this._speedColor(p.speed);
      p.point = this._points.add({
        position: ecef,
        color,
        pixelSize: this.opts.pointSize,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      });
      if (this._lines) {
        p.line = this._lines.add({
          positions: [ecef, ecef],
          width: 1,
          material: Cesium.Material.fromType("Color", { color }),
          show: false,
        });
      }
      this._particles.push(p);
    }
    return this;
  }

  // ------------------------------------------------------------------ loading

  async _loadField(fieldUrl) {
    const endsWithJson = /\.json$/i.test(fieldUrl);
    const baseDir = endsWithJson ? fieldUrl.replace(/[^/]*$/, "") : fieldUrl.replace(/\/+$/, "");
    const manifestUrl = endsWithJson ? fieldUrl : `${baseDir}/velocity-field-manifest.json`;
    const response = await fetch(manifestUrl);
    if (!response.ok) {
      throw new Error(`${manifestUrl} -> HTTP ${response.status}`);
    }
    const manifest = await response.json();
    const { nx, ny, nz } = manifest;
    if (!nx || !ny || !nz) throw new Error("velocity-field manifest missing dimensions");

    const u = new Float32Array(nx * ny * nz);
    const v = new Float32Array(nx * ny * nz);
    const w = new Float32Array(nx * ny * nz);
    const shards = manifest.shards || [];
    for (const name of shards) {
      const shardResp = await fetch(`${baseDir}${name}`);
      if (!shardResp.ok) throw new Error(`${baseDir}${name} -> HTTP ${shardResp.status}`);
      const s = await shardResp.json();
      const { zStart, zCount, yStart, yCount } = s;
      const localY = yCount;
      const su = s.u, sv = s.v, sw = s.w;
      let ptr = 0;
      for (let k = 0; k < zCount; k += 1) {
        const gk = zStart + k;
        for (let j = 0; j < yCount; j += 1) {
          const gj = yStart + j;
          for (let i = 0; i < nx; i += 1) {
            const gIdx = i + gj * nx + gk * nx * ny;
            u[gIdx] = su[ptr];
            v[gIdx] = sv[ptr];
            w[gIdx] = sw[ptr];
            ptr += 1;
          }
        }
      }
    }
    this._field = {
      nx,
      ny,
      nz,
      origin: manifest.origin,
      spacing: manifest.spacing,
      georeference: manifest.georeference,
      speedMin: manifest.speed_min,
      speedMax: manifest.speed_max,
      u,
      v,
      w,
    };
  }

  _setField(field) {
    const toF32 = (arr) =>
      arr instanceof Float32Array ? arr : Float32Array.from(arr);
    this._field = Object.assign({}, field, {
      u: toF32(field.u),
      v: toF32(field.v),
      w: toF32(field.w),
    });
  }

  // ------------------------------------------------------------- coordinate math

  _enuToEcef(enu, result) {
    const local = Cesium.Cartesian3.fromElements(enu[0], enu[1], enu[2], _scratchEnu);
    return Cesium.Matrix4.multiplyByPoint(this._frame, local, result || new Cesium.Cartesian3());
  }

  _ecefToEnu(cart, result) {
    return Cesium.Matrix4.multiplyByPoint(this._invFrame, cart, result || new Cesium.Cartesian3());
  }

  /** Trilinear velocity sample at ENU metres. Returns [u, v, w]. */
  _sampleVelocity(enu) {
    const f = this._field;
    const ix = (enu[0] - f.origin[0]) / f.spacing[0];
    const iy = (enu[1] - f.origin[1]) / f.spacing[1];
    const iz = (enu[2] - f.origin[2]) / f.spacing[2];
    const { nx, ny, nz } = f;
    const i0 = Math.min(Math.max(Math.floor(ix), 0), nx - 1);
    const j0 = Math.min(Math.max(Math.floor(iy), 0), ny - 1);
    const k0 = Math.min(Math.max(Math.floor(iz), 0), nz - 1);
    const i1 = Math.min(i0 + 1, nx - 1);
    const j1 = Math.min(j0 + 1, ny - 1);
    const k1 = Math.min(k0 + 1, nz - 1);
    const fx = Math.min(Math.max(ix - i0, 0), 1);
    const fy = Math.min(Math.max(iy - j0, 0), 1);
    const fz = Math.min(Math.max(iz - k0, 0), 1);
    const at = (arr, i, j, k) => arr[i + j * nx + k * nx * ny];
    const lerp = (a, b, t) => a + (b - a) * t;
    const c00 = lerp(at(f.u, i0, j0, k0), at(f.u, i1, j0, k0), fx);
    const c10 = lerp(at(f.u, i0, j1, k0), at(f.u, i1, j1, k0), fx);
    const c01 = lerp(at(f.u, i0, j0, k1), at(f.u, i1, j0, k1), fx);
    const c11 = lerp(at(f.u, i0, j1, k1), at(f.u, i1, j1, k1), fx);
    const u = lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz);
    const d00 = lerp(at(f.v, i0, j0, k0), at(f.v, i1, j0, k0), fx);
    const d10 = lerp(at(f.v, i0, j1, k0), at(f.v, i1, j1, k0), fx);
    const d01 = lerp(at(f.v, i0, j0, k1), at(f.v, i1, j0, k1), fx);
    const d11 = lerp(at(f.v, i0, j1, k1), at(f.v, i1, j1, k1), fx);
    const v = lerp(lerp(d00, d10, fy), lerp(d01, d11, fy), fz);
    const e00 = lerp(at(f.w, i0, j0, k0), at(f.w, i1, j0, k0), fx);
    const e10 = lerp(at(f.w, i0, j1, k0), at(f.w, i1, j1, k0), fx);
    const e01 = lerp(at(f.w, i0, j0, k1), at(f.w, i1, j0, k1), fx);
    const e11 = lerp(at(f.w, i0, j1, k1), at(f.w, i1, j1, k1), fx);
    const w = lerp(lerp(e00, e10, fy), lerp(e01, e11, fy), fz);
    return [u, v, w];
  }

  // -------------------------------------------------------------- particle life

  _seed(p) {
    const b = this._bounds;
    p.enu = [
      b.min[0] + Math.random() * (b.max[0] - b.min[0]),
      b.min[1] + Math.random() * (b.max[1] - b.min[1]),
      b.min[2] + Math.random() * (b.max[2] - b.min[2]),
    ];
    p.age = Math.random() * this.opts.maxAge; // stagger respawns
    p.history = [];
    const vel = this._sampleVelocity(p.enu);
    p.speed = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
  }

  _update() {
    if (!this._running || !this._visible) return;
    const now = (typeof performance !== "undefined" ? performance.now() : Date.now());
    const dt = this._lastTime == null ? 0 : Math.min((now - this._lastTime) / 1000, this.opts.maxDt);
    this._lastTime = now;
    if (dt <= 0) return;

    const scale = dt * this.opts.speedScale;
    const b = this._bounds;
    for (const p of this._particles) {
      const vel = this._sampleVelocity(p.enu);
      const sp = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
      p.speed = sp;
      p.age += dt;

      const shouldRespawn =
        p.age > this.opts.maxAge ||
        sp < 1e-4 ||
        Math.random() < this.opts.dropRate ||
        p.enu[0] < b.min[0] ||
        p.enu[0] > b.max[0] ||
        p.enu[1] < b.min[1] ||
        p.enu[1] > b.max[1] ||
        p.enu[2] < b.min[2] ||
        p.enu[2] > b.max[2];

      if (shouldRespawn) {
        this._seed(p);
        const ecef = this._enuToEcef(p.enu);
        p.point.position = ecef;
        p.point.color = this._speedColor(p.speed);
        if (p.line) {
          p.line.show = false;
          p.line.positions = [ecef, ecef];
        }
        continue;
      }

      p.enu[0] += vel[0] * scale;
      p.enu[1] += vel[1] * scale;
      p.enu[2] += vel[2] * scale;

      const ecef = this._enuToEcef(p.enu);
      p.point.position = ecef;
      p.point.color = this._speedColor(p.speed);

      if (p.line) {
        p.history.push(ecef);
        if (p.history.length > this.opts.trailLength) p.history.shift();
        if (p.history.length >= 2) {
          p.line.show = true;
          p.line.positions = p.history.slice();
          if (this.opts.colorBySpeed) {
            p.line.material.uniforms.color = this._speedColor(p.speed);
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------- control

  start() {
    if (this._running) return this;
    this._running = true;
    this._visible = true;
    this._lastTime = null;
    this._updateHandler = () => this._update();
    this.viewer.scene.preUpdate.addEventListener(this._updateHandler);
    return this;
  }

  stop() {
    this._running = false;
    if (this._updateHandler) {
      this.viewer.scene.preUpdate.removeEventListener(this._updateHandler);
      this._updateHandler = null;
    }
    return this;
  }

  setVisible(visible) {
    this._visible = !!visible;
    if (this._points) this._points.show = this._visible;
    if (this._lines) this._lines.show = this._visible;
    return this;
  }

  destroy() {
    this.stop();
    if (this._points && this.viewer.scene.primitives.contains(this._points)) {
      this.viewer.scene.primitives.remove(this._points);
    }
    if (this._lines && this.viewer.scene.primitives.contains(this._lines)) {
      this.viewer.scene.primitives.remove(this._lines);
    }
    this._points = null;
    this._lines = null;
    this._particles = [];
    this._field = null;
  }

  // --------------------------------------------------------------------- color

  _speedColor(speed) {
    if (!this.opts.colorBySpeed) {
      return this.opts.fixedColor || Cesium.Color.CYAN;
    }
    const f = this._field;
    const t = f.speedMax > f.speedMin
      ? Math.min(Math.max((speed - f.speedMin) / (f.speedMax - f.speedMin), 0), 1)
      : 0.5;
    // blue (slow) -> red (fast)
    const r = Math.min(1, 0.1 + t * 0.9);
    const g = Math.max(0, 0.5 - Math.abs(t - 0.5) * 1.0);
    const b = Math.min(1, 0.9 - t * 0.8);
    return new Cesium.Color(r, g, b, 0.95);
  }
}

const _scratchEnu = new Cesium.Cartesian3();

if (typeof window !== "undefined") {
  window.WindParticles = WindParticles;
}
