import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
sys.path.insert(0, "/home/claude")
import head_acoustic_mapper as ham



def element_positions(n_elements, radius_px):
    angles = np.linspace(0, 2 * np.pi, n_elements, endpoint=False)
    cx = cy = ham.GRID_SIZE / 2
    xs = cx + radius_px * np.cos(angles)
    ys = cy + radius_px * np.sin(angles)
    return np.stack([ys, xs], axis=1) 

def build_ray_system_matrix(positions, coarse_size, fine_size, n_samples=600):
  

  
    n_elements = len(positions)
    pairs = [(i, j) for i in range(n_elements) for j in range(i + 1, n_elements)]
    n_rays = len(pairs)
    n_pixels = coarse_size * coarse_size

    scale = coarse_size / fine_size
    rows, cols, vals = [], [], []
    ray_lengths_px = np.zeros(n_rays)

    for r, (i, j) in enumerate(pairs):
        y0, x0 = positions[i]
        y1, x1 = positions[j]
        length_px = np.hypot(y1 - y0, x1 - x0)
        ray_lengths_px[r] = length_px
        ts = np.linspace(0, 1, n_samples)
        ys = (y0 + ts * (y1 - y0)) * scale
        xs = (x0 + ts * (x1 - x0)) * scale
        cy = np.clip(ys.astype(int), 0, coarse_size - 1)
        cx = np.clip(xs.astype(int), 0, coarse_size - 1)
        cell_idx = cy * coarse_size + cx
        step_len = length_px / n_samples

      

        # accumulate duplicate hits on the same cell within one ray
        unique_cells, counts = np.unique(cell_idx, return_counts=True)
        rows.extend([r] * len(unique_cells))
        cols.extend(unique_cells.tolist())
        vals.extend((counts * step_len).tolist())

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_rays, n_pixels))
    return A, pairs, ray_lengths_px

  

def simulate_ray_measurements(labels, positions, pairs, freq_mhz, snr_db=None, rng=None):
    atten_img = ham.labels_to_attenuation_image(labels, freq_mhz)
    measurements = np.zeros(len(pairs))
    for r, (i, j) in enumerate(pairs):
        y0, x0 = positions[i]
        y1, x1 = positions[j]
        n_samples = 600
        ts = np.linspace(0, 1, n_samples)
        ys = np.clip((y0 + ts * (y1 - y0)).astype(int), 0, ham.GRID_SIZE - 1)
        xs = np.clip((x0 + ts * (x1 - x0)).astype(int), 0, ham.GRID_SIZE - 1)
        length_cm = np.hypot(y1 - y0, x1 - x0) * ham.PIXEL_CM
        step_cm = length_cm / n_samples
        measurements[r] = atten_img[ys, xs].sum() * step_cm  # path-integrated dB

    if snr_db is not None:
        if rng is None:
            rng = np.random.default_rng(0)
        sig_power = np.mean(measurements ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        measurements = measurements + rng.normal(0, np.sqrt(noise_power), measurements.shape)


  
    return measurements


def reconstruct_sparse_ray(A, measurements, coarse_size, damp=0.15):
    # regularized (damped) least squares: min ||Ax - b||^2 + damp^2 ||x||^2
    # damping is required because n_rays << n_pixels (massively underdetermined
    # without it) -- this is the honest fix, not a cosmetic add-on
    result = spla.lsqr(A, measurements, damp=damp, iter_lim=300)
    x = result[0]
    return x.reshape(coarse_size, coarse_size)


def downsample_truth(labels, coarse_size):
    truth_mask = labels == ham.TISSUES["plaque"]["id"]
    fine_size = labels.shape[0]
    block = fine_size // coarse_size
    coarse_truth = np.zeros((coarse_size, coarse_size), dtype=bool)
    for r in range(coarse_size):
        for c in range(coarse_size):
            patch = truth_mask[r * block:(r + 1) * block, c * block:(c + 1) * block]
            coarse_truth[r, c] = patch.mean() > 0.15
    return coarse_truth




def evaluate(recon_coarse, coarse_truth, brain_baseline):
    threshold = brain_baseline * 1.3
    pred = recon_coarse > threshold
    inter = np.logical_and(coarse_truth, pred).sum()
    dice = 2 * inter / (coarse_truth.sum() + pred.sum() + 1e-9)
    iou = inter / (np.logical_or(coarse_truth, pred).sum() + 1e-9)
    recall = inter / (coarse_truth.sum() + 1e-9)
    precision = inter / (pred.sum() + 1e-9)
    return dict(dice=dice, iou=iou, recall=recall, precision=precision)


def run_element_count_comparison(element_counts=(8, 12, 16, 24), freq_mhz=1.0,
                                  snr_db=30, n_trials=8):
    print(f"{'N elements':<12}{'N rays':<10}{'grid':<8}{'dice':<8}{'iou':<8}"
          f"{'recall':<8}{'precision':<10}")
    all_results = {}
    for n_elem in element_counts:
        n_pairs = n_elem * (n_elem - 1) // 2
        coarse_size = max(4, int(round(np.sqrt(n_pairs))))

      
        trial_metrics = []
        for trial in range(n_trials):
            rng = np.random.default_rng(2000 + trial)
            labels, plaque_masks = ham.build_phantom(rng)
            positions = element_positions(n_elem, radius_px=ham.GRID_SIZE / 2 - 3)
            A, pairs, ray_lengths = build_ray_system_matrix(
                positions, coarse_size, ham.GRID_SIZE)
            measurements = simulate_ray_measurements(
                labels, positions, pairs, freq_mhz, snr_db=snr_db, rng=rng)
            recon = reconstruct_sparse_ray(A, measurements, coarse_size)
            coarse_truth = downsample_truth(labels, coarse_size)
            baseline = ham.attenuation_db_per_cm("brain", freq_mhz)
            metrics = evaluate(recon, coarse_truth, baseline)
            trial_metrics.append(metrics)

        agg = {k: np.mean([m[k] for m in trial_metrics]) for k in trial_metrics[0]}
        all_results[n_elem] = agg
        print(f"{n_elem:<12}{n_pairs:<10}{coarse_size}x{coarse_size:<5}"
              f"{agg['dice']:<8.3f}{agg['iou']:<8.3f}{agg['recall']:<8.3f}{agg['precision']:<10.3f}")

    return all_results




def circle_intersections(y0, x0, y1, x1, cy, cx, radius):
   
    dy, dx = y1 - y0, x1 - x0
    fy, fx = y0 - cy, x0 - cx
    a = dy * dy + dx * dx
    b = 2 * (fy * dy + fx * dx)
    c = fy * fy + fx * fx - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0 or a == 0:
        return None, None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return t1, t2

  

def annulus_path_length_cm(y0, x0, y1, x1, cy, cx, r_inner_px, r_outer_px, pixel_cm):
  
    t_out1, t_out2 = circle_intersections(y0, x0, y1, x1, cy, cx, r_outer_px)
    t_in1, t_in2 = circle_intersections(y0, x0, y1, x1, cy, cx, r_inner_px)

    def clamp01(t):
        return max(0.0, min(1.0, t)) if t is not None else None

    total_len_px = np.hypot(y1 - y0, x1 - x0)
    if t_out1 is None:
        return 0.0

    outer_lo, outer_hi = sorted([clamp01(t_out1), clamp01(t_out2)])
    if t_in1 is None:
        # segment will never reache the inner circle . whole outer overlap is annulus
        annulus_frac = outer_hi - outer_lo
    else:
        inner_lo, inner_hi = sorted([clamp01(t_in1), clamp01(t_in2)])
        # annulus = (outer interval) minus (inner interval)
        annulus_frac = (outer_hi - outer_lo) - max(0.0, min(outer_hi, inner_hi) - max(outer_lo, inner_lo))

    return max(0.0, annulus_frac) * total_len_px * pixel_cm


def interior_cell_mask(coarse_size, r_inner_frac=0.84):
    cy = cx = coarse_size / 2
    yy, xx = np.mgrid[0:coarse_size, 0:coarse_size]
    r = np.hypot(yy - cy + 0.5, xx - cx + 0.5)
    return r <= (coarse_size / 2) * r_inner_frac


def build_ray_system_matrix_corrected(positions, coarse_size, fine_size, n_samples=600,
                                       r_inner_frac=0.84):
   
    n_elements = len(positions)
    pairs = [(i, j) for i in range(n_elements) for j in range(i + 1, n_elements)]
    n_rays = len(pairs)

    interior_mask = interior_cell_mask(coarse_size, r_inner_frac)
    interior_indices = {}
    idx = 0
    for r in range(coarse_size):
        for c in range(coarse_size):
            if interior_mask[r, c]:
                interior_indices[(r, c)] = idx
                idx += 1
    n_unknowns = idx

    scale = coarse_size / fine_size
    rows, cols, vals = [], [], []
    skull_lengths_cm = np.zeros(n_rays)
    cy_fine = cx_fine = fine_size / 2
    r_outer_px = fine_size / 2 - 1
    r_inner_px = r_outer_px * r_inner_frac

    for r, (i, j) in enumerate(pairs):
        y0, x0 = positions[i]
        y1, x1 = positions[j]
        skull_lengths_cm[r] = annulus_path_length_cm(
            y0, x0, y1, x1, cy_fine, cx_fine, r_inner_px, r_outer_px, ham.PIXEL_CM)

        ts = np.linspace(0, 1, n_samples)
        ys = (y0 + ts * (y1 - y0)) * scale
        xs = (x0 + ts * (x1 - x0)) * scale
        cy_s = np.clip(ys.astype(int), 0, coarse_size - 1)
        cx_s = np.clip(xs.astype(int), 0, coarse_size - 1)
        step_len = (np.hypot(y1 - y0, x1 - x0) / n_samples)

        cell_pairs = list(zip(cy_s.tolist(), cx_s.tolist()))
        seen = {}
        for cp in cell_pairs:
            if cp in interior_indices:
                seen[cp] = seen.get(cp, 0) + 1
        for cp, count in seen.items():
            rows.append(r)
            cols.append(interior_indices[cp])
            vals.append(count * step_len)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_rays, n_unknowns))
    return A, pairs, skull_lengths_cm, interior_indices, interior_mask


def select_damp_via_cross_validation_FAILED_APPROACH(A, measurements, damp_candidates, k_folds=5, rng=None):

  
    if rng is None:
        rng = np.random.default_rng(0)
    n_rays = A.shape[0]
    k = min(k_folds, max(2, n_rays // 3))

    fold_assignment = rng.integers(0, k, size=n_rays)
    best_damp, best_score = damp_candidates[0], np.inf

    for damp in damp_candidates:
        fold_errors = []
        for fold in range(k):
            held_out = fold_assignment == fold
            train = ~held_out
            if held_out.sum() == 0 or train.sum() == 0:
                continue
            x = spla.lsqr(A[train], measurements[train], damp=damp, iter_lim=300)[0]
            pred = A[held_out] @ x
            fold_errors.append(np.mean((pred - measurements[held_out]) ** 2))
        score = np.mean(fold_errors) if fold_errors else np.inf
        if score < best_score:
            best_score = score
            best_damp = damp

    return best_damp, best_score


def select_damp_via_lcurve(A, measurements, damp_candidates):
    
    from scipy.interpolate import UnivariateSpline

    log_res, log_sol = [], []
    for damp in damp_candidates:
        x = spla.lsqr(A, measurements, damp=damp, iter_lim=300)[0]
        residual_norm = np.linalg.norm(A @ x - measurements)
        solution_norm = np.linalg.norm(x)
        log_res.append(np.log(max(residual_norm, 1e-12)))
        log_sol.append(np.log(max(solution_norm, 1e-12)))

    log_res = np.array(log_res)
    log_sol = np.array(log_sol)
    log_damp = np.log10(damp_candidates)

  
    order = np.argsort(log_damp)
    log_damp_sorted = log_damp[order]
    spline_res = UnivariateSpline(log_damp_sorted, log_res[order], k=4, s=1e-3)
    spline_sol = UnivariateSpline(log_damp_sorted, log_sol[order], k=4, s=1e-3)

    d1_res = spline_res.derivative(1)(log_damp)
    d2_res = spline_res.derivative(2)(log_damp)
    d1_sol = spline_sol.derivative(1)(log_damp)
    d2_sol = spline_sol.derivative(2)(log_damp)

    curvature = np.abs(d1_res * d2_sol - d1_sol * d2_res) / \
        (d1_res ** 2 + d1_sol ** 2 + 1e-12) ** 1.5

   
    best_idx = np.argmax(curvature)
    return damp_candidates[best_idx], curvature[best_idx]



def reconstruct_with_skull_correction(labels, positions, pairs, freq_mhz, skull_lengths_cm,
                                        skull_coef_estimate, A, coarse_size, interior_indices,
                                        snr_db=None, rng=None, damp=0.15, auto_damp=False):
    raw_measurements = simulate_ray_measurements(labels, positions, pairs, freq_mhz,
                                                   snr_db=snr_db, rng=rng)
    skull_contribution = skull_coef_estimate * skull_lengths_cm
    residual = raw_measurements - skull_contribution
    residual = np.clip(residual, a_min=-np.inf, a_max=None)

    if auto_damp:
        damp_candidates = np.logspace(-3, 1.0, 20)
        damp, _ = select_damp_via_lcurve(A, residual, damp_candidates)

    x = spla.lsqr(A, residual, damp=damp, iter_lim=300)[0]
    recon_coarse = np.full((coarse_size, coarse_size), np.nan)
    for (r, c), idx in interior_indices.items():
        recon_coarse[r, c] = x[idx]
    return recon_coarse, damp


def evaluate_interior(recon_coarse, coarse_truth, brain_baseline, interior_mask):
    threshold = brain_baseline * 1.3
    valid = interior_mask
    pred = np.zeros_like(coarse_truth, dtype=bool)
    pred[valid] = recon_coarse[valid] > threshold
    truth = coarse_truth & valid
    inter = np.logical_and(truth, pred).sum()
    dice = 2 * inter / (truth.sum() + pred.sum() + 1e-9)
    iou = inter / (np.logical_or(truth, pred).sum() + 1e-9)
    recall = inter / (truth.sum() + 1e-9)
    precision = inter / (pred.sum() + 1e-9)
    return dict(dice=dice, iou=iou, recall=recall, precision=precision)


def run_corrected_comparison(element_counts=(8, 12, 16, 24), freq_mhz=1.0,
                              snr_db=30, n_trials=8, skull_calib_error_pct=10,
                              auto_damp=True, fixed_coarse_size=12):
   
                                
    print(f"{'N elements':<12}{'N rays':<10}{'grid':<8}{'dice':<8}{'iou':<8}"
          f"{'recall':<8}{'precision':<10}{'damp(mean)':<12}")
    all_results = {}
    true_skull_coef = ham.attenuation_db_per_cm("skull", freq_mhz)
    coarse_size = fixed_coarse_size

    for n_elem in element_counts:
        n_pairs = n_elem * (n_elem - 1) // 2

        trial_metrics = []
        damps_used = []
        for trial in range(n_trials):
            rng = np.random.default_rng(3000 + trial)
            labels, plaque_masks = ham.build_phantom(rng)
            positions = element_positions(n_elem, radius_px=ham.GRID_SIZE / 2 - 3)
            A, pairs, skull_lengths, interior_indices, interior_mask = \
                build_ray_system_matrix_corrected(positions, coarse_size, ham.GRID_SIZE)

            calib_noise = rng.normal(1.0, skull_calib_error_pct / 100.0)
            skull_coef_estimate = true_skull_coef * calib_noise

            recon, damp_used = reconstruct_with_skull_correction(
                labels, positions, pairs, freq_mhz, skull_lengths,
                skull_coef_estimate, A, coarse_size, interior_indices,
                snr_db=snr_db, rng=rng, auto_damp=auto_damp)
            damps_used.append(damp_used)

            coarse_truth = downsample_truth(labels, coarse_size)
            baseline = ham.attenuation_db_per_cm("brain", freq_mhz)
            recon_filled = np.nan_to_num(recon, nan=-999)
            metrics = evaluate_interior(recon_filled, coarse_truth, baseline, interior_mask)
            trial_metrics.append(metrics)

        agg = {k: np.mean([m[k] for m in trial_metrics]) for k in trial_metrics[0]}
        agg["damp_mean"] = float(np.mean(damps_used))
        agg["damp_std"] = float(np.std(damps_used))
        all_results[n_elem] = agg
        print(f"{n_elem:<12}{n_pairs:<10}{coarse_size}x{coarse_size:<5}"
              f"{agg['dice']:<8.3f}{agg['iou']:<8.3f}{agg['recall']:<8.3f}{agg['precision']:<10.3f}"
              f"{agg['damp_mean']:<12.4f}")

    return all_results


def reconstruct_with_skull_correction_sparse_prior(labels, positions, pairs, freq_mhz,
                                                     skull_lengths_cm, skull_coef_estimate,
                                                     A, coarse_size, interior_indices,
                                                     snr_db=None, rng=None, alpha=0.001):

    from sklearn.linear_model import Lasso

    raw_measurements = simulate_ray_measurements(labels, positions, pairs, freq_mhz,
                                                   snr_db=snr_db, rng=rng)
    skull_contribution = skull_coef_estimate * skull_lengths_cm
    residual = np.clip(raw_measurements - skull_contribution, a_min=0, a_max=None)

    lasso = Lasso(alpha=alpha, positive=True, max_iter=10000)
    lasso.fit(A, residual)
    x = lasso.coef_

    recon_coarse = np.zeros((coarse_size, coarse_size))
    for (r, c), idx in interior_indices.items():
        recon_coarse[r, c] = x[idx]
    return recon_coarse


def evaluate_interior_relative_threshold(recon_coarse, coarse_truth, interior_mask, frac=0.35):
   
    valid_vals = recon_coarse[interior_mask]
    max_val = valid_vals.max() if valid_vals.size and valid_vals.max() > 0 else None
    pred = np.zeros_like(recon_coarse, dtype=bool)
    if max_val is not None:
        pred[interior_mask] = recon_coarse[interior_mask] > frac * max_val

    truth = coarse_truth & interior_mask
    inter = np.logical_and(truth, pred).sum()
    dice = 2 * inter / (truth.sum() + pred.sum() + 1e-9)
    iou = inter / (np.logical_or(truth, pred).sum() + 1e-9)
    recall = inter / (truth.sum() + 1e-9)
    precision = inter / (pred.sum() + 1e-9)
    return dict(dice=dice, iou=iou, recall=recall, precision=precision)


def detect_and_trace_sparse_ray(recon, interior_mask, coarse_size, fine_pixel_cm,
                                 fine_grid_size, upsample_factor=6, min_component_px=2):
   
    from skimage.filters import threshold_otsu
    from skimage import measure
    from scipy import ndimage

    vals = recon[interior_mask]
    if vals.max() <= 0:
        return []
    try:
        t = threshold_otsu(vals)
    except Exception:
        t = vals.max() * 0.35

    mask = np.zeros_like(recon, dtype=bool)
    mask[interior_mask] = recon[interior_mask] > t

    upsampled_mask = ndimage.zoom(mask.astype(float), upsample_factor, order=1) > 0.5
    upsampled_recon = ndimage.zoom(np.nan_to_num(recon, nan=0.0), upsample_factor, order=1)

    labeled, n = ndimage.label(upsampled_mask)
    coarse_cell_cm = (fine_grid_size * fine_pixel_cm) / coarse_size
    up_pixel_cm = coarse_cell_cm / upsample_factor
    up_size = coarse_size * upsample_factor

    detections = []
    for i in range(1, n + 1):
        comp_mask = labeled == i
        if comp_mask.sum() < min_component_px:
            continue
        contours = measure.find_contours(comp_mask.astype(float), level=0.5)
        if not contours:
            continue
        outline = max(contours, key=len)
        outline_cm = [
            (round((c - up_size / 2) * up_pixel_cm, 3), round((r - up_size / 2) * up_pixel_cm, 3))
            for r, c in outline[::2]
        ]
        ys, xs = np.where(comp_mask)
        cy, cx = ys.mean(), xs.mean()
        detections.append({
            "centroid_x_cm": round((cx - up_size / 2) * up_pixel_cm, 3),
            "centroid_y_cm": round((cy - up_size / 2) * up_pixel_cm, 3),
            "pixel_count_upsampled": int(comp_mask.sum()),
            "peak_value": round(float(upsampled_recon[comp_mask].max()), 4),
            "traced_outline_cm": outline_cm,
            "outline_px_upsampled": [(float(r), float(c)) for r, c in outline[::2]],
        })
    return detections


def run_sparse_prior_comparison(element_counts=(8, 12, 16, 24), freq_mhz=1.0,
                                 snr_db=30, n_trials=8, skull_calib_error_pct=10,
                                 fixed_coarse_size=12, alpha=0.001):
    print(f"{'N elements':<12}{'N rays':<10}{'grid':<8}{'dice':<8}{'iou':<8}"
          f"{'recall':<8}{'precision':<10}")
    all_results = {}
    true_skull_coef = ham.attenuation_db_per_cm("skull", freq_mhz)
    coarse_size = fixed_coarse_size

    for n_elem in element_counts:
        n_pairs = n_elem * (n_elem - 1) // 2
        trial_metrics = []
        for trial in range(n_trials):
            rng = np.random.default_rng(3000 + trial)
            labels, plaque_masks = ham.build_phantom(rng)
            positions = element_positions(n_elem, radius_px=ham.GRID_SIZE / 2 - 3)
            A, pairs, skull_lengths, interior_indices, interior_mask = \
                build_ray_system_matrix_corrected(positions, coarse_size, ham.GRID_SIZE)

            calib_noise = rng.normal(1.0, skull_calib_error_pct / 100.0)
            skull_coef_estimate = true_skull_coef * calib_noise

            recon = reconstruct_with_skull_correction_sparse_prior(
                labels, positions, pairs, freq_mhz, skull_lengths,
                skull_coef_estimate, A, coarse_size, interior_indices,
                snr_db=snr_db, rng=rng, alpha=alpha)

            coarse_truth = downsample_truth(labels, coarse_size)
            metrics = evaluate_interior_relative_threshold(recon, coarse_truth, interior_mask)
            trial_metrics.append(metrics)

        agg = {k: np.mean([m[k] for m in trial_metrics]) for k in trial_metrics[0]}
        all_results[n_elem] = agg
        print(f"{n_elem:<12}{n_pairs:<10}{coarse_size}x{coarse_size:<5}"
              f"{agg['dice']:<8.3f}{agg['iou']:<8.3f}{agg['recall']:<8.3f}{agg['precision']:<10.3f}")

    return all_results


if __name__ == "__main__":
    print("=== Attempt 1: naive joint solve (no skull correction) ===")
    print("(corner-biased garbage -- demonstrates the original bug)")
    naive_results = run_element_count_comparison(snr_db=30, n_trials=5)

    print("\n=== Attempt 2: skull correction + L2/Tikhonov, fixed grid=12 for fair comparison ===")
    print("(isolates element-count effect; still expect instability -- see diagnosis)")
    l2_results = run_corrected_comparison(snr_db=30, n_trials=6, auto_damp=True,
                                           fixed_coarse_size=12)

    print("\n=== Attempt 3 (FINAL): L1/sparse-prior reconstruction, relative threshold ===")
    print("(matches the true 'sparse cluster in empty background' structure)")
    final_results_30 = run_sparse_prior_comparison(snr_db=30, n_trials=8)
    print("\n--- Same, at poorer SNR=15dB ---")
    final_results_15 = run_sparse_prior_comparison(snr_db=15, n_trials=8)
    print("\n--- Same, with worse skull calibration error (25% instead of 10%) ---")
    final_results_badcal = run_sparse_prior_comparison(snr_db=30, n_trials=8,
                                                         skull_calib_error_pct=25)

    import json
    with open("/mnt/user-data/outputs/sparse_ray_tomography_results.json", "w") as f:
        json.dump({
            "attempt1_naive_uncorrected": {str(k): v for k, v in naive_results.items()},
            "attempt2_L2_tikhonov_fixed_grid": {str(k): v for k, v in l2_results.items()},
            "attempt3_FINAL_L1_sparse_prior_snr30": {str(k): v for k, v in final_results_30.items()},
            "attempt3_FINAL_L1_sparse_prior_snr15": {str(k): v for k, v in final_results_15.items()},
            "attempt3_FINAL_L1_sparse_prior_snr30_badcalib25pct": {str(k): v for k, v in final_results_badcal.items()},
        }, f, indent=2, default=float)
    print("\nsaved sparse_ray_tomography_results.json")
