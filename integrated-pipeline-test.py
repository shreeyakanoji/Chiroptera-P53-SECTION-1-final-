import numpy as np
import sys
import head_acoustic_mapper as ham
import sparse_ray_tomography as srt
import headset_calibration as cal


print("=== Step 1: run bench calibration (as would happen on real hardware) ===")
sweep = cal.run_gain_voltage_sweep()
best = cal.select_calibrated_settings(sweep)
calib = cal.calibrate_skull_coefficient(freq_mhz=1.0, drive_voltage=best["voltage"],
                                          gain_db=best["gain_db"], length_range_cm=(1.0, 3.0))
print(f"\ncalibrated skull coefficient: {calib['estimated_skull_coef_db_cm']:.3f} "
      f"+/- {calib['uncertainty_std']:.3f} dB/cm "
      f"({calib['relative_error_pct']:.1f}% error vs ground truth)")

print("\n=== Step 2: feed the CALIBRATED (not assumed) coefficient into reconstruction ===")


def run_with_real_calibration(calibrated_coef, calibrated_uncertainty, element_counts=(8, 12, 16, 24),
                               freq_mhz=1.0, snr_db=30, n_trials=8):
    print(f"{'N elements':<12}{'N rays':<10}{'grid':<8}{'dice':<8}{'iou':<8}"
          f"{'recall':<8}{'precision':<10}")
    all_results = {}
    for n_elem in element_counts:
        n_pairs = n_elem * (n_elem - 1) // 2
        coarse_size = max(6, int(round(np.sqrt(n_pairs))) + 2)

        trial_metrics = []
        for trial in range(n_trials):
            rng = np.random.default_rng(3000 + trial)
            labels, plaque_masks = ham.build_phantom(rng)
            positions = srt.element_positions(n_elem, radius_px=ham.GRID_SIZE / 2 - 3)
            A, pairs, skull_lengths, interior_indices, interior_mask = \
                srt.build_ray_system_matrix_corrected(positions, coarse_size, ham.GRID_SIZE)

          
            skull_coef_estimate = rng.normal(calibrated_coef, calibrated_uncertainty)

            recon = srt.reconstruct_with_skull_correction(
                labels, positions, pairs, freq_mhz, skull_lengths,
                skull_coef_estimate, A, coarse_size, interior_indices,
                snr_db=snr_db, rng=rng)

            coarse_truth = srt.downsample_truth(labels, coarse_size)
            baseline = ham.attenuation_db_per_cm("brain", freq_mhz)
            recon_filled = np.nan_to_num(recon, nan=-999)
            metrics = srt.evaluate_interior(recon_filled, coarse_truth, baseline, interior_mask)
            trial_metrics.append(metrics)

        agg = {k: np.mean([m[k] for m in trial_metrics]) for k in trial_metrics[0]}
        all_results[n_elem] = agg
        print(f"{n_elem:<12}{n_pairs:<10}{coarse_size}x{coarse_size:<5}"
              f"{agg['dice']:<8.3f}{agg['iou']:<8.3f}{agg['recall']:<8.3f}{agg['precision']:<10.3f}")
    return all_results


results = run_with_real_calibration(calib["estimated_skull_coef_db_cm"], calib["uncertainty_std"])

import json
with open("outputs/integrated_pipeline_results.json", "w") as f:
    json.dump({
        "calibration_step": calib,
        "reconstruction_with_real_calibration": {str(k): v for k, v in results.items()},
    }, f, indent=2, default=float)
print("\nsaved integrated_pipeline_results.json -- this is the full calibration -> reconstruction loop")
