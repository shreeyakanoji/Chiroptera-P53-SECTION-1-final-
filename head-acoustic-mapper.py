import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import radon, iradon
from skimage import measure
from scipy import ndimage
import plotly.graph_objects as go
import json



GRID_SIZE = 220
HEAD_RADIUS_CM = 8.5
PIXEL_CM = (2 * HEAD_RADIUS_CM) / GRID_SIZE

TISSUES = {
    "air":    {"id": 0, "a": 0.0,  "b": 0.0,   "speed": 343,  "color": "#e8f4f8"},
    "skin":   {"id": 1, "a": 0.6,  "b": 0.02,  "speed": 1540, "color": "#e8b48a"},
    "skull":  {"id": 2, "a": 7.8,  "b": 0.35,  "speed": 2800, "color": "#f2ead8"},
    "csf":    {"id": 3, "a": 0.05, "b": 0.001, "speed": 1500, "color": "#a8d8f0"},
    "brain":  {"id": 4, "a": 0.58, "b": 0.008, "speed": 1560, "color": "#d9a3c9"},
    "target": {"id": 5, "a": 0.58, "b": 0.008, "speed": 1560, "color": "#c98fb5"},
    "plaque": {"id": 6, "a": 1.35, "b": 0.05,  "speed": 1620, "color": "#8b1a1a"},
}
ID_TO_NAME = {v["id"]: k for k, v in TISSUES.items()}


def attenuation_db_per_cm(tissue_name, freq_mhz):
    t = TISSUES[tissue_name]
    return t["a"] * freq_mhz + t["b"] * freq_mhz ** 2



def build_phantom_from_simnibs(rng, slice_index=128, margin_px=8):
    import nibabel as nib
    nii_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "simnibs_data", "m2m_ernie", "final_tissues.nii.gz")
    img = nib.load(nii_path)
    vol = img.get_fdata()[..., 0]
    raw_slice = vol[:, slice_index, :]  

    ys, xs = np.where(raw_slice > 0)
    y0, y1 = max(ys.min() - margin_px, 0), min(ys.max() + margin_px, raw_slice.shape[0])
    x0, x1 = max(xs.min() - margin_px, 0), min(xs.max() + margin_px, raw_slice.shape[1])
    cropped = raw_slice[y0:y1, x0:x1]

    global GRID_SIZE, PIXEL_CM
    GRID_SIZE = int(max(cropped.shape))
    PIXEL_CM = 0.1  # 1mm isotropic voxels 

    padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=cropped.dtype)
    oy = (GRID_SIZE - cropped.shape[0]) // 2
    ox = (GRID_SIZE - cropped.shape[1]) // 2
    padded[oy:oy + cropped.shape[0], ox:ox + cropped.shape[1]] = cropped


  
    sim_to_tissue = {
        0: "air", 1: "brain", 2: "brain", 3: "csf",
        4: "skull", 5: "skin", 6: "skin", 7: "skull", 8: "skull",
        9: "skin", 10: "skin",
    }
    labels = np.zeros_like(padded, dtype=np.uint8)
    for sim_id, tissue_name in sim_to_tissue.items():
        labels[padded == sim_id] = TISSUES[tissue_name]["id"]


  
    gm_mask = labels == TISSUES["brain"]["id"]
    gy, gx = np.where(gm_mask)
    cy_brain, cx_brain = gy.mean(), gx.mean()
    col_idx = np.arange(GRID_SIZE)[None, :]
    row_idx = np.arange(GRID_SIZE)[:, None]
    lower_medial = gm_mask & (row_idx > cy_brain) \
        & (np.abs(col_idx - cx_brain) < GRID_SIZE * 0.22) \
        & (np.abs(col_idx - cx_brain) > GRID_SIZE * 0.06)
    labels[lower_medial] = TISSUES["target"]["id"]
    target_mask = labels == TISSUES["target"]["id"]

    ys_t, xs_t = np.where(target_mask)
    plaque_masks = []
    if len(ys_t) > 0:
        for _ in range(5):
            idx = rng.integers(0, len(ys_t))
            seed = (float(ys_t[idx]), float(xs_t[idx]))
            mask = grow_fibrillar_plaque(seed, target_mask, rng)
            if mask.sum() >= 4:
                labels[mask] = TISSUES["plaque"]["id"]
                plaque_masks.append(mask)

    return labels, plaque_masks



def build_phantom(rng):
    y, x = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    cy, cx = GRID_SIZE / 2, GRID_SIZE / 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    R = GRID_SIZE / 2

    labels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    labels[r <= R] = TISSUES["skin"]["id"]
    labels[r <= R * 0.96] = TISSUES["skull"]["id"]
    labels[r <= R * 0.88] = TISSUES["csf"]["id"]
    labels[r <= R * 0.84] = TISSUES["brain"]["id"]

  
    tx, ty = cx, cy - R * 0.35
    ellipse = (((x - tx) / (R * 0.22)) ** 2 + ((y - ty) / (R * 0.14)) ** 2) <= 1
    labels[ellipse & (r <= R * 0.84)] = TISSUES["target"]["id"]

  
    target_mask = labels == TISSUES["target"]["id"]
    ys, xs = np.where(target_mask)
    n_plaques = 5
    plaque_masks = []
    for _ in range(n_plaques):
        idx = rng.integers(0, len(ys))
        seed = (float(ys[idx]), float(xs[idx]))
        mask = grow_fibrillar_plaque(seed, target_mask, rng)
        if mask.sum() < 4:
            continue
        labels[mask] = TISSUES["plaque"]["id"]
        plaque_masks.append(mask)

    return labels, plaque_masks


def grow_fibrillar_plaque(seed, region_mask, rng, n_branches=None, branch_len=None):
    h, w = region_mask.shape
    shape_mask = np.zeros((h, w), dtype=bool)
    if n_branches is None:
        n_branches = rng.integers(3, 6)
    if branch_len is None:
        branch_len = rng.integers(14, 26)

    def stamp(py, px, r=1):
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - py) ** 2 + (xx - px) ** 2 <= r ** 2
        shape_mask[blob & region_mask] = True

    stamp(*seed, r=2)
    for _ in range(n_branches):
        pos = np.array(seed, dtype=float)
        angle = rng.uniform(0, 2 * np.pi)
        for step in range(branch_len):
            angle += rng.normal(0, 0.35)
            pos = pos + np.array([np.sin(angle), np.cos(angle)])
            py, px = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= py < h and 0 <= px < w) or not region_mask[py, px]:
                break
            stamp(py, px, r=1)


          
            if step > 4 and rng.random() < 0.12:
                fork_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.3)
                fpos = pos.copy()
                for fstep in range(max(3, branch_len // 3)):
                    fork_angle += rng.normal(0, 0.3)
                    fpos = fpos + np.array([np.sin(fork_angle), np.cos(fork_angle)])
                    fpy, fpx = int(round(fpos[0])), int(round(fpos[1]))
                    if not (0 <= fpy < h and 0 <= fpx < w) or not region_mask[fpy, fpx]:
                        break
                    stamp(fpy, fpx, r=1)
    return shape_mask


def labels_to_attenuation_image(labels, freq_mhz):
    img = np.zeros(labels.shape, dtype=np.float64)
    for name, t in TISSUES.items():
        img[labels == t["id"]] = attenuation_db_per_cm(name, freq_mhz)
    return img



def acquire_sinogram(atten_img_db_cm, n_angles=90):
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(atten_img_db_cm, theta=angles, circle=True)
    sinogram *= PIXEL_CM 
  
    return sinogram, angles


def reconstruct(sinogram, angles):
    recon = iradon(sinogram / PIXEL_CM, theta=angles, circle=True, filter_name="ramp")
    return recon


def detect_plaques(recon_img, brain_baseline_db_cm, margin=0.15):


  
    threshold = brain_baseline_db_cm * (1 + margin) * 1.4
    mask = recon_img > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    labeled, n = ndimage.label(mask)
    props = measure.regionprops(labeled, intensity_image=recon_img)

    detections = []
    for p in props:
        if p.area < 3:
            continue
        region_mask = labeled == p.label
        contours = measure.find_contours(region_mask.astype(float), level=0.5)
        if not contours:
            continue
        outline = max(contours, key=len)  # outer boundary
        outline_cm = [
            (round((c - GRID_SIZE / 2) * PIXEL_CM, 3), round((r - GRID_SIZE / 2) * PIXEL_CM, 3))
            for r, c in outline[::2]  
        ]
        cy, cx = p.centroid
        detections.append({
            "row": float(cy),
            "col": float(cx),
            "centroid_x_cm": round((cx - GRID_SIZE / 2) * PIXEL_CM, 3),
            "centroid_y_cm": round((cy - GRID_SIZE / 2) * PIXEL_CM, 3),
            "pixel_count": int(p.area),
            "peak_attenuation_db_cm": round(float(p.intensity_max), 3),
            "perimeter_px": round(float(p.perimeter), 2),
            "eccentricity": round(float(p.eccentricity), 3),  # 0=circular, ->1 elongated/irregular
            "traced_outline_cm": outline_cm,
            "outline_px": [(float(r), float(c)) for r, c in outline[::2]],
        })
    return detections



def frequency_sweep(labels, freqs_mhz, n_angles=90):
    results = []
    brain_baseline = attenuation_db_per_cm("brain", 1.0)
    for f in freqs_mhz:
        atten_img = labels_to_attenuation_image(labels, f)
        sino, angles = acquire_sinogram(atten_img, n_angles=n_angles)
        recon = reconstruct(sino, angles)
        baseline_f = attenuation_db_per_cm("brain", f)
        detections = detect_plaques(recon, baseline_f)
        skull_atten = attenuation_db_per_cm("skull", f)
        results.append({
            "freq_mhz": f,
            "atten_img": atten_img,
            "sinogram": sino,
            "recon": recon,
            "detections": detections,
            "skull_atten_db_cm": skull_atten,
        })
    return results




def plot_pipeline(labels, sweep_results, plaque_masks_truth, out_path):
    n_freqs = len(sweep_results)
    fig = plt.figure(figsize=(4.2 * (n_freqs + 1), 9))
    gs = GridSpec(2, n_freqs + 1, figure=fig)

    cmap_labels = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    for name, t in TISSUES.items():
        mask = labels == t["id"]
        rgb = np.array([int(t["color"][1:3], 16), int(t["color"][3:5], 16), int(t["color"][5:7], 16)]) / 255
        cmap_labels[mask] = rgb

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(cmap_labels)
    ax0.set_title("Ground-truth tissue phantom\n(headset ring plane)")
    ax0.axis("off")
    for pm in plaque_masks_truth:
        for contour in measure.find_contours(pm.astype(float), level=0.5):
            ax0.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=1.3)

    ax0b = fig.add_subplot(gs[1, 0])
    ax0b.axis("off")
    legend_text = "\n".join([f"{name}: a={t['a']}, b={t['b']} (dB/cm, dB/cm/MHz^2)" for name, t in TISSUES.items()])
    ax0b.text(0.0, 0.5, legend_text, fontsize=8, va="center", family="monospace")
    ax0b.set_title("Attenuation model per tissue")

    for i, res in enumerate(sweep_results):
        col = i + 1
        ax_recon = fig.add_subplot(gs[0, col])
        im = ax_recon.imshow(res["recon"], cmap="inferno")
        ax_recon.set_title(f"Reconstructed map — traced outlines\n{res['freq_mhz']} MHz")
        ax_recon.axis("off")
        for d in res["detections"]:
            outline = np.array(d["outline_px"])
            ax_recon.plot(outline[:, 1], outline[:, 0], color="cyan", linewidth=1.3)
        plt.colorbar(im, ax=ax_recon, fraction=0.046, label="dB/cm")

        ax_sino = fig.add_subplot(gs[1, col])
        ax_sino.imshow(res["sinogram"], cmap="viridis", aspect="auto")
        ax_sino.set_title(f"Sinogram (raw headset signal)\nskull loss: {res['skull_atten_db_cm']:.2f} dB/cm")
        ax_sino.set_xlabel("transducer angle index")
        ax_sino.set_ylabel("ray offset (px)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




def plot_frequency_response(out_path):
    freqs = np.linspace(0.1, 3.0, 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, t in TISSUES.items():
        if name == "air":
            continue
        vals = [attenuation_db_per_cm(name, f) for f in freqs]
        ax.plot(freqs, vals, label=name, linewidth=2 if name in ("brain", "plaque") else 1)
    ax.set_xlabel("Headset drive frequency (MHz)")
    ax.set_ylabel("Attenuation (dB/cm)")
    ax.set_title("Frequency vs tissue attenuation\n(higher freq = better contrast, worse skull penetration)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def grow_fibrillar_plaque_3d(seed_xyz, rng, n_branches=5, branch_len=16, step=0.05):
    branches = []
    for _ in range(n_branches):
        pos = np.array(seed_xyz, dtype=float)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pts = [pos.copy()]
        for s in range(branch_len):
            direction += rng.normal(scale=0.45, size=3)
            direction /= np.linalg.norm(direction)
            pos = pos + direction * step
            pts.append(pos.copy())
            if s > 3 and rng.random() < 0.15:
                fdir = direction + rng.normal(scale=0.8, size=3)
                fdir /= np.linalg.norm(fdir)
                fpos = pos.copy()
                fpts = [fpos.copy()]
                for fs in range(max(3, branch_len // 3)):
                    fdir += rng.normal(scale=0.4, size=3)
                    fdir /= np.linalg.norm(fdir)
                    fpos = fpos + fdir * step
                    fpts.append(fpos.copy())
                branches.append(np.array(fpts))
        branches.append(np.array(pts))
    return branches


def hemisphere_surface(lateral_extent, length_half, height_half, side, n=90,
                        wrinkle_amp=0.025, seed=3):
  
    rng = np.random.default_rng(seed + (1 if side > 0 else 2))
    u = np.linspace(-np.pi / 2, np.pi / 2, n)  
    v = np.linspace(0.001, np.pi - 0.001, n)    
    uu, vv = np.meshgrid(u, v, indexing="ij")

    r = np.ones_like(uu)
    for k in range(5):
        fu = rng.uniform(4, 11)
        fv = rng.uniform(4, 11)
        amp = wrinkle_amp / (k + 1) ** 0.7
        pu = rng.uniform(0, 2 * np.pi)
        pv = rng.uniform(0, 2 * np.pi)
        r += amp * np.sin(fu * uu + pu) * np.cos(fv * vv + pv)

    x_local = lateral_extent * r * np.cos(uu) * np.sin(vv)  
    y_raw = length_half * np.sin(uu) * np.sin(vv)            
    z_raw = height_half * np.cos(vv)                         

    y_scale = np.where(y_raw > 0, 1.08, 0.90)
    y_final = y_raw * y_scale
    z_scale = np.where(z_raw > 0, 1.0, 0.72)
    z_final = z_raw * z_scale
    return x_local, y_final, z_final


def small_lobe_surface(a, b, c, center, n=30, seed=1):
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0.001, np.pi - 0.001, n)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    r = np.ones_like(uu)
    for k in range(3):
        fu, fv = rng.uniform(3, 7), rng.uniform(3, 7)
        amp = 0.03 / (k + 1)
        r += amp * np.sin(fu * uu) * np.cos(fv * vv)
    x = center[0] + a * r * np.cos(uu) * np.sin(vv)
    y = center[1] + b * r * np.sin(uu) * np.sin(vv)
    z = center[2] + c * r * np.cos(vv)
    return x, y, z

STL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_stl")

CATEGORY_STYLE = {
    "hippocampus":  {"color": "#ff7a5c", "opacity": 0.95},
    "amygdala":     {"color": "#ffb84d", "opacity": 0.9},
    "cerebellum":   {"color": "#3f8f6a", "opacity": 0.55},
    "thalamus":     {"color": "#8a6acb", "opacity": 0.7},
    "ventric":      {"color": "#6ab0d8", "opacity": 0.18},
    "gyrus":        {"color": "#d9b98a", "opacity": 0.28},
    "lobe":         {"color": "#d9b98a", "opacity": 0.22},
    "insula":       {"color": "#d9b98a", "opacity": 0.28},
    "cingulate":    {"color": "#c98fb5", "opacity": 0.4},
    "pons":         {"color": "#8a8a6a", "opacity": 0.6},
    "medulla":      {"color": "#8a8a6a", "opacity": 0.6},
    "midbrain":     {"color": "#8a8a6a", "opacity": 0.6},
    "peduncle":     {"color": "#8a8a6a", "opacity": 0.5},
    "colliculus":   {"color": "#8a8a6a", "opacity": 0.5},
    "brachium":     {"color": "#8a8a6a", "opacity": 0.4},
    "putamen":      {"color": "#7a6aa8", "opacity": 0.6},
    "caudate":      {"color": "#7a6aa8", "opacity": 0.6},
    "globus pallidus": {"color": "#7a6aa8", "opacity": 0.6},
    "geniculate":   {"color": "#8a8a6a", "opacity": 0.5},
}
DEFAULT_STYLE = {"color": "#c9c9c9", "opacity": 0.35}
MAX_TRIANGLES_PER_MESH = 4000




def style_for(name):
    name_l = name.lower()
    for key, style in CATEGORY_STYLE.items():
        if key in name_l:
            return style
    return DEFAULT_STYLE


def load_fma_names():
    names = {}
    names_path = os.path.join(os.path.dirname(STL_DIR), "parts_list_e.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    names[parts[0]] = parts[1]
    return names


def decimate_triangles(pts, max_tris):
    n = pts.shape[0]
    if n <= max_tris:
        return pts
    stride = int(np.ceil(n / max_tris))
    return pts[::stride]




def build_3d_head(out_path, n_plaque_clusters=4):
    from stl import mesh as stl_mesh

    rng = np.random.default_rng(11)
    names = load_fma_names()
    stl_files = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))

    fig = go.Figure()
    all_pts = []
    hippo_centroids = []

    for path in stl_files:
        fid = os.path.splitext(os.path.basename(path))[0]
        label = names.get(fid, fid)
        m = stl_mesh.Mesh.from_file(path)
        pts_mm = decimate_triangles(m.points, MAX_TRIANGLES_PER_MESH).reshape(-1, 3)
        all_pts.append(pts_mm)
        if "hippocampus" in label.lower() and "para" not in label.lower():
            hippo_centroids.append(pts_mm.mean(axis=0))

    all_pts_concat = np.vstack(all_pts)
    origin_mm = all_pts_concat.mean(axis=0)  # recenter the whole brain at (0,0,0)

    for path, pts_mm in zip(stl_files, all_pts):
        fid = os.path.splitext(os.path.basename(path))[0]
        label = names.get(fid, fid)
        style = style_for(label)

        pts_cm = (pts_mm - origin_mm) / 10.0
        n_tri = pts_cm.shape[0] // 3
        pts_cm = pts_cm[: n_tri * 3]
        idx = np.arange(n_tri * 3).reshape(n_tri, 3)

        fig.add_trace(go.Mesh3d(
            x=pts_cm[:, 0], y=pts_cm[:, 1], z=pts_cm[:, 2],
            i=idx[:, 0], j=idx[:, 1], k=idx[:, 2],
            color=style["color"], opacity=style["opacity"],
            name=label, hoverinfo="name",
            flatshading=True,
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.3, roughness=0.6),
            lightposition=dict(x=100, y=-100, z=150),
            showlegend=False,
        ))


    if hippo_centroids:
        target_center = (np.mean(hippo_centroids, axis=0) - origin_mm) / 10.0
    else:
        target_center = np.array([0.0, -9.0, -1.5])

  
    cluster_colors = ["#ff3b1f", "#ff9500", "#ffe14d", "#ff5ba8"]
    marker_points = []
    marker_customdata = []
    for cluster_i in range(n_plaque_clusters):
        offset = rng.normal(scale=[0.35, 0.4, 0.3])
        seed = target_center + offset
        marker_points.append(seed)
        side_label = "right" if seed[0] > 0 else "left"
        color = cluster_colors[cluster_i % len(cluster_colors)]
        branches = grow_fibrillar_plaque_3d(seed, rng, n_branches=rng.integers(4, 7),
                                             branch_len=rng.integers(8, 14), step=0.035)
        n_branches = len(branches)
        total_pts = sum(len(b) for b in branches)
        info_html = (f"<b>Plaque cluster {cluster_i + 1}</b><br>"
                      f"Location: {side_label} hippocampus, "
                      f"({seed[0]:.2f}, {seed[1]:.2f}, {seed[2]:.2f}) cm<br>"
                      f"Fibril branches: {n_branches}<br>"
                      f"Traced points: {total_pts}")
        marker_customdata.append(info_html)

        for b_i, branch in enumerate(branches):
            fig.add_trace(go.Scatter3d(
                x=branch[:, 0], y=branch[:, 1], z=branch[:, 2],
                mode="lines",
                line=dict(color=color, width=10),
                opacity=1.0,
                name=f"plaque cluster {cluster_i + 1}",
                customdata=[info_html] * len(branch),
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=(b_i == 0),
            ))


  
    marker_points = np.array(marker_points)
    fig.add_trace(go.Scatter3d(
        x=marker_points[:, 0], y=marker_points[:, 1], z=marker_points[:, 2],
        mode="markers",
        marker=dict(size=7, color=cluster_colors[:len(marker_points)], opacity=1.0,
                    line=dict(color="white", width=2)),
        name="plaque cluster site",
        customdata=marker_customdata,
        hovertemplate="%{customdata}<extra></extra>",
    ))

  
    fig.update_layout(
        title=dict(
            text="Real anatomical brain mesh (BodyParts3D) — click a plaque cluster for details",
            font=dict(color="white", size=15),
        ),
        paper_bgcolor="black",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="black", aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.5, z=0.7)),
            dragmode="orbit",
        ),
        legend=dict(font=dict(color="white")),
        clickmode="event+select",
        annotations=[dict(
            text="Mesh data: BodyParts3D, (c) Database Center for Life Science, CC BY-SA 2.1 Japan"
                 "  |  scroll to zoom, drag to rotate, click a plaque for details",
            x=0, y=0, xref="paper", yref="paper", showarrow=False,
            font=dict(color="#888", size=10),
        )],
        width=1000, height=880,
    )

    click_js = """
    var infoBox = document.createElement('div');
    infoBox.id = 'plaque-info-box';
    infoBox.style.cssText = 'position:fixed;bottom:24px;left:24px;background:#111;color:#fff;'
        + 'padding:14px 18px;border-radius:10px;font-family:-apple-system,sans-serif;'
        + 'font-size:13px;line-height:1.5;max-width:320px;display:none;'
        + 'border:1px solid #555;z-index:1000;box-shadow:0 4px 20px rgba(0,0,0,0.5);';
    document.body.appendChild(infoBox);

    var closeBtn = document.createElement('span');
    closeBtn.innerHTML = ' &times;';
    closeBtn.style.cssText = 'float:right;cursor:pointer;color:#aaa;font-weight:bold;';
    closeBtn.onclick = function(){ infoBox.style.display = 'none'; };

    var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
    plotDiv.on('plotly_click', function(data){
        var pt = data.points[0];
        var content = pt.customdata ? pt.customdata : ('<b>' + pt.data.name + '</b>');
        infoBox.innerHTML = content;
        infoBox.appendChild(closeBtn);
        infoBox.style.display = 'block';
    });
    """

    fig.write_html(
        out_path,
        config={"scrollZoom": True, "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"]},
        post_script=click_js,
    )




def main():
    rng = np.random.default_rng(7)
    use_real_anatomy = os.environ.get("USE_SIMNIBS", "1") == "1"
    if use_real_anatomy:
        labels, plaque_masks_truth = build_phantom_from_simnibs(rng)
        out_prefix = "outputs/real_anatomy_"
    else:
        labels, plaque_masks_truth = build_phantom(rng)
        out_prefix = "outputs/synthetic_"

  
    freqs_mhz = [0.5, 1.0, 2.0]
    sweep_results = frequency_sweep(labels, freqs_mhz)

    plot_pipeline(labels, sweep_results, plaque_masks_truth, out_prefix + "headset_mapping_pipeline.png")
    plot_frequency_response(out_prefix + "tissue_frequency_response.png")
    build_3d_head("outputs/head_3d_interactive.html")

    summary = {
        "phantom_source": "SimNIBS Ernie (real MRI segmentation)" if use_real_anatomy else "synthetic ellipse phantom",
        "ground_truth_plaque_count": len(plaque_masks_truth),
        "pixel_cm": PIXEL_CM,
        "sweeps": [
            {
                "freq_mhz": r["freq_mhz"],
                "skull_atten_db_cm": round(r["skull_atten_db_cm"], 3),
                "n_detected": len(r["detections"]),
                "detections": [
                    {k: v for k, v in d.items() if k != "outline_px"}
                    for d in r["detections"]
                ],
            }
            for r in sweep_results
        ],
    }
    with open(out_prefix + "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for r in sweep_results:
        print(f"{r['freq_mhz']} MHz -> skull loss {r['skull_atten_db_cm']:.2f} dB/cm, "
              f"{len(r['detections'])} plaque cluster(s) traced "
              f"(ground truth: {len(plaque_masks_truth)})")


if __name__ == "__main__":
    main()




def add_measurement_noise(sinogram, snr_db, rng):
    signal_power = np.mean(sinogram ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), sinogram.shape)
    return sinogram + noise


def run_multi_phantom_validation(n_trials=20, freq_mhz=1.0, n_angles=90, snr_db=None):
    results = []
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        labels, plaque_masks = build_phantom(rng)
        atten_img = labels_to_attenuation_image(labels, freq_mhz)
        sino, angles = acquire_sinogram(atten_img, n_angles=n_angles)
        if snr_db is not None:
            sino = add_measurement_noise(sino, snr_db, rng)
        recon = reconstruct(sino, angles)

        truth_mask = labels == TISSUES["plaque"]["id"]
        threshold = attenuation_db_per_cm("brain", freq_mhz) * 1.15 * 1.4
        pred_mask = recon > threshold
        intersection = np.logical_and(truth_mask, pred_mask).sum()
        dice = 2 * intersection / (truth_mask.sum() + pred_mask.sum() + 1e-9)
        iou = intersection / (np.logical_or(truth_mask, pred_mask).sum() + 1e-9)
        recall = intersection / (truth_mask.sum() + 1e-9)
        precision = intersection / (pred_mask.sum() + 1e-9)


        from skimage.transform import iradon as _iradon
        naive_recon = _iradon(sino / PIXEL_CM, theta=angles, circle=True, filter_name=None)
        naive_pred = naive_recon > threshold
        naive_inter = np.logical_and(truth_mask, naive_pred).sum()
        naive_dice = 2 * naive_inter / (truth_mask.sum() + naive_pred.sum() + 1e-9)

        results.append({
            "trial": trial, "n_true_clusters": len(plaque_masks),
            "dice": dice, "iou": iou, "recall": recall, "precision": precision,
            "naive_dice": naive_dice,
        })
    return results


def summarize_validation(results, label):
    keys = ["dice", "iou", "recall", "precision", "naive_dice"]
    print(f"\n--- {label} (n={len(results)} random phantoms) ---")
    for k in keys:
        vals = np.array([r[k] for r in results])
        print(f"{k:12s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")


if __name__ == "__main__" and os.environ.get("RUN_MULTI_VALIDATION"):
    clean = run_multi_phantom_validation(n_trials=20, freq_mhz=1.0, snr_db=None)
    summarize_validation(clean, "Clean signal (no measurement noise)")

    noisy_30 = run_multi_phantom_validation(n_trials=20, freq_mhz=1.0, snr_db=30)
    summarize_validation(noisy_30, "Realistic noise, SNR=30dB")

    noisy_15 = run_multi_phantom_validation(n_trials=20, freq_mhz=1.0, snr_db=15)
    summarize_validation(noisy_15, "Poor noise, SNR=15dB")

  
    with open("outputs/multi_phantom_validation.json", "w") as f:
        json.dump({
            "clean": clean, "snr_30db": noisy_30, "snr_15db": noisy_15,
        }, f, indent=2, default=float)


