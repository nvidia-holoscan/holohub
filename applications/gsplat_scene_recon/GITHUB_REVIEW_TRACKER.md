# G-SHARP PR – GitHub review comments tracker

Temporary file to track review issues and fixes. **Keep in .gitignore – do not commit.**

---

## Summary from main review (G-SHARP v0.2)

**Context:** PR introduces G-SHARP v0.2: end-to-end surgical scene reconstruction (five phases: DA2+MedSAM3 → VGGT poses → EndoNeRF conversion → deformable GSplat training → live viewer), orchestrated by `run_gsharp.py` in a single Docker image.

**Reviewer verdict:** Architecture and components are clear, but several correctness issues span the stack. **Confidence: 2/5. Not safe to merge** without addressing: missing Dockerfile deps, empty-mask NaN, monocular depth path mismatch, hardcoded near/far overwrite.

---

## Comment difficulty breakdown

| Difficulty | Count | Description |
|------------|-------|-------------|
| **Easy** | **19** | Single-file or config/docs change, one-liners, simple guards; low risk (&lt;~30 min each). |
| **Moderate** | **22** | Multi-file or non-trivial logic, well-scoped; typically 30–90 min each. |
| **Difficult** | **11** | Cross-component, subtle correctness, or behavior-changing logic; required careful reasoning and/or full pipeline runs to verify. |
| **Open** | **1** | Dead threshold-scheduling code (#6); not yet addressed. |
| **Total addressed** | **52** | (Some issue IDs are duplicates or same fix, e.g. #50 = #21.) |

*Easy examples:* metadata version/tags/paths (#10, #27, #28, #29), MedSAM3 `_torch_stream` init (#4), `weights_only=True` (#5), Dockerfile docs/guards (#23, #24, #26, #35, #36), render_viewer fps/TRAINING_DIR (#38, #39), assert→exception (#41, #45, #47).
*Moderate examples:* Pin deps (#11), USER/chown (#12), device fixes (#13, #14), depth_utils validation (#15), empty-mask NaN (#3), RMSE normalization (#30), cache/percentile/downsample/num_pts (#32, #33, #40, #42, #43), MONITOR move (#52), format_conversion reruns/masks (#46, #48), l1_loss 4D mask (#51).
*Difficult examples:* DataLoader coarse skip (#1), near/far from Phase 3 (#2), VGGT vggt_hw + near floor (#9), mask polarity (#16), depth_supervise_tools + monocular weight (#17, #18), behind-camera Gaussians (#19), multi-frame initializer with poses (#20), checkpoint config + deformation_table save/load (#22, #34), deformation table after pruning (#31), SCARED cx/cy (#44).

---

## 1. Docker fixes

| # | Issue | Location | Status | Fix applied |
|---|--------|----------|--------|-------------|
| 8 | **Missing fpsample and open3d in Dockerfile** – Both imported by `endo_loader.py`; Phase 4 ImportError. | Dockerfile pip install | 🟢 Addressed | Added `fpsample` and `open3d` to pip install block (pinned versions). |
| 11 | **Pin Python deps** – Floating versions make rebuilds non-reproducible. | Dockerfile 35–54 | 🟢 Addressed | requirements.txt with pinned versions; Dockerfile COPY + pip install -r requirements.txt. |
| 12 | **Drop privileges in Dockerfile** – No USER after install; app runs as root. | Dockerfile 85–108 | 🟢 Addressed | Added `appuser` (UID 1000), chown /workspace; USER appuser; cache path docs. |
| 23 | **Dockerfile: runtime mount docs** – Header says only assets/data/; example uses output and cache. | Dockerfile 7–12, 21–24, 106–108 | 🟢 Addressed | Docs list assets/, data/, output/, optional cache. |
| 24 | **Dockerfile: build context** – `docker build .` assumes caller in app dir. | Dockerfile 13–15 | 🟢 Addressed | Added "run from applications/gsplat_scene_recon or use -f". |
| 25 | **Dockerfile: x86_64 lib path** – decord workaround hardcodes /usr/lib/x86_64-linux-gnu; no-op on aarch64. | Dockerfile 64–69 | 🟢 Addressed | Use dpkg-architecture -q DEB_HOST_MULTIARCH with fallback; comment notes x86_64 primary. |
| 26 | **VGG16 hash verification** – load_state_dict_from_url doesn't verify hash. | Dockerfile 84–85 | 🟢 Addressed | Added check_hash=True. |
| 35 | **Dockerfile: guard libxcb loop against empty glob** – No libxcb*.so* → loop variable/basename misbehaves. | Dockerfile 49–52 | 🟢 Addressed | Add `[ -e "$f" ] \|\| continue` inside loop. |
| 36 | **Dockerfile: vggt/sam3 transitive deps** – pip install git+... resolves deps not in requirements.txt; build drift. | Dockerfile 62–64 | 🟢 Addressed | Added --no-deps to pip install; comment to add missing dep to requirements.txt if import fails. |

---

## 2. Gsplat training fixes

| # | Issue | Location | Status | Fix applied |
|---|--------|----------|--------|-------------|
| 1 | **DataLoader result discarded every coarse-stage step** – next(trainloader_iter) then overwritten with self.trainset[0]; unnecessary disk I/O. | gsplat_train.py 1037–1049 | 🟢 Addressed | Coarse: get data from self.trainset[0] only; fine: use DataLoader as before. |
| 2 | **Hardcoded near/far bounds in train_standalone.py** – accumulate_data() rebuilds poses_bounds.npy with [0.01, 1000.0], discarding Phase 3 VGGT near/far. | train_standalone.py line 140 | 🟢 Addressed | Keep bounds_src from initial load; write bounds_src[:frames_to_process] into saved poses_bounds.npy. |
| 3 | **Silent NaN from empty-mask l1_loss** – mask all zeros → loss[mask!=0].mean() is NaN. | loss_utils.py line 134 | 🟢 Addressed | Raise ValueError when mask is not None and masked loss is empty. |
| 6 | **Dead threshold-scheduling code** – Interpolated threshold assigned to _and never applied to step_post_backward. | gsplat_train.py 1323–1331 | 🔴 Open | — |
| 7 | **Monocular depth mode non-functional** – Loader expects monodepth/; format_conversion only creates depth/. | endo_loader.py line 142 vs format_conversion.py | 🟢 Addressed | Documented pipeline outputs binocular only; raise clear ValueError when monocular selected but monodepth/ missing. |
| 9 | **format_conversion: VGGT resolution 518 hardcoded; near-plane 0 when min VGGT depth zero** | format_conversion.py, vggt_inference.py | 🟢 Addressed | VGGT saves vggt_hw.npy; format_conversion loads it (fallback 518); near = max(near, 1.0). |
| 16 | **Mask polarity inconsistent** – 464–465/2079–2082 use 1=tissue/0=tool; 1926–1933 invert again; one path backwards. | gsplat_train.py 464–465, 1921–1935, 2079–2082 | 🟢 Addressed | Convention: raw PNG 0=tissue, 255=tool. create_invisible_mask uses raw/255 as tool (no second invert); **getitem** comment. |
| 17 | **depth_supervise_tools broken** – Zero depth_gt when masks enabled; monocular branch dereferences mask → crash. | gsplat_train.py 938–951, 1063–1066, 1179–1192 | 🟢 Addressed | When depth_supervise_tools=True do not zero depth_gt; mask=None guard in monocular path. |
| 18 | **Monocular depth weighted twice** – Monocular returns pearson_lambda*(1-corr); total loss multiplies by depth_lambda again. | gsplat_train.py 950–951, 1274–1277 | 🟢 Addressed | Monocular returns (1-corr); caller uses depth_weight = pearson_lambda if monocular else depth_lambda. |
| 19 | **Behind-camera Gaussians in tool mask** – project_gaussians clamps z to 1e-6; points behind camera counted as tool hits. | gsplat_train.py 1666–1695, 1723–1738 | 🟢 Addressed | project_gaussians_to_image returns (means_2d, in_front); check_gaussians_in_tool_region ANDs with valid_mask. |
| 20 | **Multi-frame initializer ignores camera motion** – Merges all frames into one buffer, unprojects once with K; no poses. | gsplat_train.py 2057–2137 | 🟢 Addressed | Per-frame: get_color_depth_mask → get_pts_cam → get_pts_wld(pose); concat; downsample to 200k. |
| 21 | **AttributeError when mask is None (monocular)** – compute_depth_loss monocular path mask.reshape(-1) without guard. | gsplat_train.py 938–951 | 🟢 Addressed | Guard: if mask is not None use masked tensors else use full. |
| 22 | **Checkpoint missing EndoConfig** – _save_checkpoint doesn't save config; viewer uses hardcoded defaults. | gsplat_train.py 1458–1478, render_viewer.py 86–111 | 🟢 Addressed | Save config dict in checkpoint; viewer uses NS(**ckpt["config"]) when present. |
| 30 | **RMSE normalization wrong for non-2D** – Unmasked branch divides by a.shape[-1]*a.shape[-2] only; wrong for [H,W,C]. | image_utils.py 135–147 | 🟢 Addressed | Use a.size for unmasked RMSE. |
| 31 | **Deformation table index mismatch after pruning** – Pruning compacts indices; truncating _deformation_table misaligns table vs splats. | gsplat_train.py 1357–1374 | 🟢 Addressed | On shrink: reset table to ones(num_gs_after), accum to zeros; tool masking re-applies on next update. |
| 32 | **Dataset cache never populated** – EndoNeRFDataset._cache read in **getitem** but never written. | gsplat_train.py 419–436 | 🟢 Addressed | Populate cache after building data: self._cache[frame_idx] = data before return. |
| 33 | **np.percentile raises on all-zero binocular depth** – depth PNG all zeros → depth[depth!=0] empty → ValueError. | gsplat_train.py 450–454 | 🟢 Addressed | Guard: if nonzero.size == 0 use zeros; else percentile + clip. |
| 34 | **_deformation_table not saved — viewer deforms all Gaussians** – enable_tool_masking=True; checkpoint didn't save table; viewer passed all N through deform net. | gsplat_train.py 1457–1497, render_viewer.py 125–142 | 🟢 Addressed | Save data["deformation_table"] in_save_checkpoint; load in_load_checkpoint;_apply_deformation only pass deformable subset. |
| 40 | **endo_loader: downsample vs full-res tensors** – img_wh/focal scaled but RGB/depth/mask full-res; get_pts_cam wrong when downsample!=1. | endo_loader.py 105–116, 196–197, 284–291 | 🟢 Addressed | Reject downsample!=1.0 in load_meta() with ValueError; pipeline uses 1.0. |
| 42 | **endo_loader: guard empty depth before percentile** – depth[depth!=0] empty → np.percentile/min/max raise. | endo_loader.py 181–185, 270–274, 431–438 | 🟢 Addressed | Guards in get_color_depth_mask, format_infos binocular branch, SCARED bounds. |
| 43 | **endo_loader: num_pts==0 in get_init_pts** – sample_count=1 and np.random.choice(0,...) raises. | endo_loader.py 234–241, 549–556, 575–577 | 🟢 Addressed | Skip frame when num_pts==0; if pts_total empty raise RuntimeError; hgi_mono raises if num_pts==0. |
| 44 | **endo_loader: SCARED cx/cy dropped** – Full KL in JSON but rays use W/2, H/2; off-center principal point wrong. | endo_loader.py 381–383, 449–450, 593–600 | 🟢 Addressed | SCARED load_meta sets self.cx, self.cy from camera_mat; get_pts_cam uses getattr(self,'cx',W/2). |
| 49 | **gsplat_train: all-zero depth in another path** – Multi-frame init uses endo_dataset.get_color_depth_mask. | gsplat_train.py 2087–2092 | 🟢 Addressed | Fixed via #42 in endo_loader.get_color_depth_mask. |
| 50 | **AttributeError when mask is None (monocular)** – Same as #21. | gsplat_train.py 938–951 | 🟢 Addressed | Already fixed in #21. |
| 51 | **l1_loss 4D mask batch dimension** – mask [1,1,H,W] or [1,C,H,W]; loss [B,C,H,W] B>1 → loss[mask!=0] fail. | loss_utils.py 96–132 | 🟢 Addressed | Expand batch when mask.shape[0]==1 and loss.shape[0]>1 before channel expansion. |
| 52 | **[MONITOR] gradient norms always 0** – Block ran before loss.backward(); param.grad None. Also raw print() spam. | gsplat_train.py 1109–1165 | 🟢 Addressed | Moved MONITOR block to after loss.backward() and before optimizer.step(); converted to logger.debug(). |

---

## 3. Code polish fixes

| # | Issue | Location | Status | Fix applied |
|---|--------|----------|--------|-------------|
| 4 | **_torch_stream not initialised in MedSAM3SegmentationOp.**init**** – Only set in start(); compute() before start() → AttributeError. | medsam3_segmentation_op.py line 38 | 🟢 Addressed | Initialised _torch_stream = None in **init**. |
| 5 | **torch.load with weights_only=False** – Disables pickle safety; arbitrary code from tampered checkpoint. | depth_anything_v2_op.py, sam3_inference.py, render_viewer.py | 🟢 Addressed | weights_only=True at all three call sites. |
| 10 | **Version field inconsistent with PR title** – metadata.json "version": "1.0" vs PR G-SHARP v0.2. | metadata.json | 🟢 Addressed | Set "version": "0.2" (later 0.2.0 for semver). |
| 13 | **mask_token device in dinov2.py** – .to(x.dtype) keeps original device; CPU/GPU mismatch in torch.where. | dinov2.py 233–234 | 🟢 Addressed | mask_token.to(dtype=x.dtype, device=x.device). |
| 14 | **dpt.py: image to model device** – DEVICE sends input to first accelerator; model on CPU → device mismatch. | dpt.py 260–265 | 🟢 Addressed | image = image.to(next(self.parameters()).device) in infer_image. |
| 15 | **sparse_depths_to_dense validation** – No validation of shapes, method, or tracks_2d [N,2]. | depth_utils.py 16–23, 40–44, 61 | 🟢 Addressed | ValueError for shape/length/method/height/width. |
| 27 | **metadata semver** – version and changelog should be major.minor.patch; release 0.2.0. | metadata.json 12–16 | 🟢 Addressed | version "0.2.0", changelog key "0.2.0" only. |
| 28 | **First tag approved category** – "3D Reconstruction" not approved; move e.g. "Healthcare AI" to index 0. | metadata.json 27–35 | 🟢 Addressed | "Healthcare AI" first. |
| 29 | **run config build-relative path** – run.command/workdir mix holohub_app_source; align with holohub_bin. | metadata.json 146–148 | 🟢 Addressed | workdir=holohub_app_bin; command uses run_gsharp.py with --data-dir/--output-dir. |
| 37 | **vggt_inference: overlap=0 alignment** – When overlap=0, ref and new are different frames; alignment assumes same frame. | vggt_inference.py 258–265 | 🟢 Addressed | Document in comment; overlap>=1 recommended for multi-batch. |
| 38 | **render_viewer: deformation import without TRAINING_DIR** – scene.deformation fails unless TRAINING_DIR set. | render_viewer.py 84–110, 348–351 | 🟢 Addressed | Default TRAINING_DIR to SCRIPT_DIR / "training". |
| 39 | **render_viewer: reject non-positive fps** – fps=0 or negative → div by fps, sleep logic crash. | render_viewer.py 181–190 | 🟢 Addressed | In GsplatRenderOp.**init**, raise ValueError if fps <= 0. |
| 41 | **endo_loader: assert → explicit validation** – assert stripped by python -O. | endo_loader.py 155–163 | 🟢 Addressed | Replaced 3 asserts with if/raise ValueError. |
| 45 | **format_conversion: assert → exception** – DA2 file count check; -O strips assert. | format_conversion.py 174–176 | 🟢 Addressed | if len(da2_files) != N: raise RuntimeError(...). |
| 46 | **format_conversion: reruns leave stale frames** – Shorter rerun leaves old files in images/depth/masks. | format_conversion.py 188–190, 215–245 | 🟢 Addressed | Before writing, remove existing *.png in out/images, out/depth, out/masks. |
| 47 | **format_conversion: progress import path** – from stages.progress fails when run as python format_conversion.py. | format_conversion.py 202–205 | 🟢 Addressed | try: from stages.progress import ... except ImportError: from progress import .... |
| 48 | **format_conversion: don't silently drop missing masks** – Missing mask still reports success for N masks. | format_conversion.py 230–245 | 🟢 Addressed | If mask_src does not exist, raise FileNotFoundError with frame index. |

---

## Application execution summary

Runs were performed to verify fixes and avoid regressions. Approximate counts:

| Run type | Approximate count | When / purpose |
|----------|-------------------|----------------|
| **Full E2E pipeline** (all 5 phases, with or without visualization) | **6–8** | After Docker dep fixes; after train_standalone/DataLoader fixes; after empty-mask NaN and loss_utils; after RMSE normalization; after mask/rendered alignment and dimension normalization in gsplat_train; after deformation_table save/load; after clearing phase4_training (permission fix); final verification. |
| **verify_train** (10 frames, train-only, headless) | **3–5** | After metadata.json argument names (--training-iterations); after adding opencv/tensorboard to requirements; after various gsplat_train shape/rasterization fixes. |
| **holohub test gsplat_scene_recon** | **3–4** | After CMakeLists.txt path/MedSAM3 download fix; after test_all_imports; confirmation that tests pass. |
| **Docker build** (full or no-cache) | **4–5** | After adding fpsample/open3d; after pinning requirements; after Dockerfile USER/docs; after multiarch decord fix. |

**Total application runs (build + test + full/verify_train): ~20–25** across the review cycle. Most full-pipeline and verify_train runs followed a batch of Docker, gsplat training, or code polish fixes to confirm no regressions.

**Review items by difficulty:** 19 easy, 22 moderate, 11 difficult (52 addressed total); 1 open (#6).

---

## Notes

**Docker run-as-host-UID:** Using `-u $(id -u):$(id -g)` was tried for VGGT cache write permission but caused `KeyError: getpwuid(): uid not found` (PyTorch/torchvision use getpass.getuser()). Reverted; E2E runs as image user (appuser). For cache writes, ensure host `~/.cache/huggingface` is writable by UID 1000 or host UID is 1000.

**Needs separate test (if revalidating):** #16 (mask polarity), #20 (multi-frame initializer), #12 (USER/mounts) were validated during the above runs; re-test if those code paths change.

---

*Last updated: Tracker reorganized into Docker / gsplat training / code polish; execution summary added.*
