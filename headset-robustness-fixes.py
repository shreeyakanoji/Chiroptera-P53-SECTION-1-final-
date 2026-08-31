import numpy as np
import sys
import head_acoustic_mapper as ham
import sparse_ray_tomography as srt
import headset_calibration as cal

SOUND_SPEED_TISSUE_CM_PER_US = 0.154   # ~1540 m/s i.e  standard soft-tissue value




def simulate_raw_waveform(path_length_cm, true_echo_amplitude, sample_rate_mhz=20,
                           window_us=None, clutter_amplitude=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    expected_tof_us = path_length_cm / SOUND_SPEED_TISSUE_CM_PER_US
    if window_us is None:
        window_us = expected_tof_us * 1.6 + 5  
    n_samples = int(window_us * sample_rate_mhz)
    t_us = np.linspace(0, window_us, n_samples)
    waveform = rng.normal(0, 3.0, n_samples)  # electronic noise floor


                             
    echo_idx = int(expected_tof_us * sample_rate_mhz)
    pulse_width_samples = max(3, int(0.5 * sample_rate_mhz))
    if 0 <= echo_idx < n_samples:
        for i in range(max(0, echo_idx - pulse_width_samples), min(n_samples, echo_idx + pulse_width_samples)):
            waveform[i] += true_echo_amplitude * np.exp(-0.5 * ((i - echo_idx) / (pulse_width_samples / 2)) ** 2)

                             
    if clutter_amplitude > 0:
        clutter_idx = rng.integers(0, n_samples)
        for i in range(max(0, clutter_idx - pulse_width_samples), min(n_samples, clutter_idx + pulse_width_samples)):
            waveform[i] += clutter_amplitude * np.exp(-0.5 * ((i - clutter_idx) / (pulse_width_samples / 2)) ** 2)

    return waveform, t_us, expected_tof_us


def extract_amplitude_blind(waveform):
    return np.max(np.abs(waveform))




def extract_amplitude_tof_windowed(waveform, t_us, expected_tof_us, margin_us=1.5):
    mask = np.abs(t_us - expected_tof_us) <= margin_us
    if not mask.any():
        return 0.0
    return np.max(np.abs(waveform[mask]))


def test_tof_windowing(n_trials=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    blind_errors, windowed_errors = [], []
    for _ in range(n_trials):
        path_length = rng.uniform(2, 8)
        true_amp = rng.uniform(20, 80)
        clutter_amp = rng.uniform(30, 100)  
        waveform, t_us, tof = simulate_raw_waveform(path_length, true_amp,
                                                     clutter_amplitude=clutter_amp, rng=rng)
        blind = extract_amplitude_blind(waveform)
        windowed = extract_amplitude_tof_windowed(waveform, t_us, tof)
        blind_errors.append(abs(blind - true_amp))
        windowed_errors.append(abs(windowed - true_amp))
    return dict(
        blind_mean_error=np.mean(blind_errors),
        windowed_mean_error=np.mean(windowed_errors),
        improvement_factor=np.mean(blind_errors) / max(np.mean(windowed_errors), 1e-9),
    )




def run_coupling_check(n_elements, bad_elements=None, tolerance_frac=0.5,
                        drive_voltage=60, rng=None):
    if rng is None:
        rng = np.random.default_rng(2)
    bad_elements = set(bad_elements or [])

   
    ref_cap, _, _ = cal.simulate_bench_capture(
        2.0, drive_voltage, 20, ham.attenuation_db_per_cm("skin", 1.0), rng=rng)
    expected_range = (ref_cap * (1 - tolerance_frac), ref_cap * (1 + tolerance_frac))

    results = []
    for elem in range(n_elements):
        coupling_efficiency = 0.15 if elem in bad_elements else rng.uniform(0.85, 1.0)
        cap, sat, noise_rms = cal.simulate_bench_capture(
            2.0, drive_voltage, 20, ham.attenuation_db_per_cm("skin", 1.0), rng=rng)
        measured = cap * coupling_efficiency
        ok = expected_range[0] <= measured <= expected_range[1]
        results.append(dict(element=elem, measured_amplitude=float(measured), ok=ok))

    flagged = [r["element"] for r in results if not r["ok"]]
    return results, flagged, expected_range




def select_confirmatory_rays(A, interior_indices, coarse_size, recon, n_confirm=8):
    # this is smarter than random ray selection b/c it picks rays that pass
    # through the coarse cells the scan flaggs as abnormal, since
    # those are exactly the measurements a real motion artifact would
    # corrupt. Falls back to random selection if nothing was flagged.
    flat_recon = recon.flatten()
    if flat_recon.max() <= 0:
        return None
    threshold = 0.35 * flat_recon.max()
    flagged_coarse_idx = set(np.where(flat_recon > threshold)[0].tolist())
    if not flagged_coarse_idx:
        return None

    idx_to_coarse = {v: k for k, v in interior_indices.items()}
    ray_scores = []
    A_csr = A.tocsr()
    for r in range(A.shape[0]):
        cols = A_csr.indices[A_csr.indptr[r]:A_csr.indptr[r + 1]]
        hits = sum(1 for c in cols if idx_to_coarse.get(c, (None, None))[0] is not None
                   and (idx_to_coarse[c][0] * coarse_size + idx_to_coarse[c][1]) in flagged_coarse_idx)
        if hits > 0:
            ray_scores.append((r, hits))

    ray_scores.sort(key=lambda x: -x[1])
    return [r for r, _ in ray_scores[:n_confirm]] if ray_scores else None


def run_motion_check(n_elements=12, snr_db=30, motion_occurred=False, n_repeat_rays=8, rng=None,
                       targeted=False, A=None, interior_indices=None, coarse_size=None, recon=None,
                       n_averages=1):
    if rng is None:
        rng = np.random.default_rng(3)

    labels_a, _ = ham.build_phantom(np.random.default_rng(5000))
    if motion_occurred:
        labels_b, _ = ham.build_phantom(np.random.default_rng(5001))
    else:
        labels_b = labels_a

    positions = srt.element_positions(n_elements, radius_px=ham.GRID_SIZE / 2 - 3)
    pairs = [(i, j) for i in range(n_elements) for j in range(i + 1, n_elements)]

    if targeted and A is not None:
        targeted_ray_indices = select_confirmatory_rays(A, interior_indices, coarse_size, recon, n_repeat_rays)
        if targeted_ray_indices:
            repeat_pairs = [pairs[i] for i in targeted_ray_indices]
        else:
            repeat_pairs = [pairs[i] for i in np.random.default_rng(4).choice(len(pairs), n_repeat_rays, replace=False)]
    else:
        repeat_pairs = [pairs[i] for i in np.random.default_rng(4).choice(len(pairs), n_repeat_rays, replace=False)]

    if n_averages > 1:
        first_pass = simulate_ray_measurements_averaged(labels_a, positions, repeat_pairs, 1.0, snr_db, n_averages, rng)
        second_pass = simulate_ray_measurements_averaged(labels_b, positions, repeat_pairs, 1.0, snr_db, n_averages, rng)
        effective_snr_db = snr_db + 10 * np.log10(n_averages)  # averaging K times improves SNR by 10*log10(K) dB
    else:
        first_pass = srt.simulate_ray_measurements(labels_a, positions, repeat_pairs, 1.0, snr_db=snr_db, rng=rng)
        second_pass = srt.simulate_ray_measurements(labels_b, positions, repeat_pairs, 1.0, snr_db=snr_db, rng=rng)
        effective_snr_db = snr_db

    diffs = np.abs(first_pass - second_pass)

    clean_reference = srt.simulate_ray_measurements(labels_a, positions, repeat_pairs, 1.0,
                                                       snr_db=None, rng=rng)
    sig_power = np.mean(clean_reference ** 2)
    noise_std = np.sqrt(sig_power / (10 ** (effective_snr_db / 10)))
    expected_diff_std = np.sqrt(2) * noise_std
    threshold = 5 * expected_diff_std

    flagged = diffs > threshold

    return dict(
        mean_diff=float(diffs.mean()),
        max_diff=float(diffs.max()),
        expected_noise_only_diff_std=float(expected_diff_std),
        threshold=float(threshold),
        n_flagged=int(flagged.sum()),
        n_checked=len(repeat_pairs),
        n_averages=n_averages,
        motion_detected=bool(flagged.sum() >= 1),  
    )




FDA_TRACK3_MI_LIMIT = 1.9
FDA_TRACK3_ISPTA_LIMIT_MW_CM2 = 720.0


def estimate_dosimetry(drive_voltage_vpp, freq_mhz, burst_cycles, pulse_repetition_rate_hz,
                        assumed_pressure_per_volt_kpa=10.0):
   
    peak_pressure_kpa = (drive_voltage_vpp / 2) * assumed_pressure_per_volt_kpa
    peak_pressure_mpa = peak_pressure_kpa / 1000.0

    mechanical_index = peak_pressure_mpa / np.sqrt(freq_mhz)

    burst_duration_s = burst_cycles / (freq_mhz * 1e6)
    duty_cycle = burst_duration_s * pulse_repetition_rate_hz

    rho_c_rayl = 1.5e6  # tissue characteristic acoustic impedance = Pa*s/m
    intensity_w_cm2 = (peak_pressure_mpa * 1e6) ** 2 / (2 * rho_c_rayl) / 1e4
    ispta_mw_cm2 = intensity_w_cm2 * duty_cycle * 1000

    return dict(
        assumed_pressure_per_volt_kpa=assumed_pressure_per_volt_kpa,
        peak_pressure_mpa=round(peak_pressure_mpa, 3),
        mechanical_index=round(mechanical_index, 3),
        mi_limit=FDA_TRACK3_MI_LIMIT,
        mi_pass=mechanical_index <= FDA_TRACK3_MI_LIMIT,
        duty_cycle=round(duty_cycle, 6),
        ispta_mw_cm2=round(ispta_mw_cm2, 2),
        ispta_limit_mw_cm2=FDA_TRACK3_ISPTA_LIMIT_MW_CM2,
        ispta_pass=ispta_mw_cm2 <= FDA_TRACK3_ISPTA_LIMIT_MW_CM2,
    )





def simulate_ray_measurements_averaged(labels, positions, pairs, freq_mhz, snr_db, n_averages, rng):
    accum = np.zeros(len(pairs))
    for _ in range(n_averages):
        accum += srt.simulate_ray_measurements(labels, positions, pairs, freq_mhz, snr_db=snr_db, rng=rng)
    return accum / n_averages


def run_averaging_comparison(n_elem=16, freq_mhz=1.0, snr_db=15, n_trials=8,
                              average_counts=(1, 4, 16, 64), fixed_coarse_size=12):
    print(f"{'n_averages':<12}{'scan_time_factor':<18}{'dice':<8}{'recall':<8}{'precision':<10}")
    true_skull_coef = ham.attenuation_db_per_cm("skull", freq_mhz)
    results = {}

    for n_avg in average_counts:
        trial_metrics = []
        for trial in range(n_trials):
            rng = np.random.default_rng(6000 + trial)
            labels, plaque_masks = ham.build_phantom(rng)
            positions = srt.element_positions(n_elem, radius_px=ham.GRID_SIZE / 2 - 3)
            A, pairs, skull_lengths, interior_indices, interior_mask = \
                srt.build_ray_system_matrix_corrected(positions, fixed_coarse_size, ham.GRID_SIZE)

            raw = simulate_ray_measurements_averaged(labels, positions, pairs, freq_mhz, snr_db, n_avg, rng)
            calib_noise = rng.normal(1.0, 0.10)
            skull_est = true_skull_coef * calib_noise
            residual = np.clip(raw - skull_est * skull_lengths, 0, None)

          
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.001, positive=True, max_iter=10000)
            lasso.fit(A, residual)
            recon = np.zeros((fixed_coarse_size, fixed_coarse_size))
            for (r, c), idx in interior_indices.items():
                recon[r, c] = lasso.coef_[idx]

            coarse_truth = srt.downsample_truth(labels, fixed_coarse_size)
            metrics = srt.evaluate_interior_relative_threshold(recon, coarse_truth, interior_mask)
            trial_metrics.append(metrics)

        agg = {k: np.mean([m[k] for m in trial_metrics]) for k in trial_metrics[0]}
        results[n_avg] = agg
        print(f"{n_avg:<12}{n_avg:<18}{agg['dice']:<8.3f}{agg['recall']:<8.3f}{agg['precision']:<10.3f}")

    return results


if __name__ == "__main__":
    print("=== Fix #3: time-of-flight windowed capture vs blind peak-picking ===")
    tof_result = test_tof_windowing()
    print(f"blind peak-picking mean error:    {tof_result['blind_mean_error']:.1f}")
    print(f"ToF-windowed mean error:          {tof_result['windowed_mean_error']:.1f}")
    print(f"improvement factor:               {tof_result['improvement_factor']:.1f}x")

    print("\n=== Fix #4: per-element coupling quality self-test ===")
    coupling_results, flagged, expected_range = run_coupling_check(n_elements=12, bad_elements=[3, 7])
    print(f"expected good-coupling range (derived from reference measurement): "
          f"{expected_range[0]:.0f}-{expected_range[1]:.0f}")
    for r in coupling_results:
        flag = " <-- FLAGGED (poor coupling)" if not r["ok"] else ""
        print(f"  element {r['element']:>2}: amplitude={r['measured_amplitude']:.1f}{flag}")
    print(f"correctly flagged elements: {flagged} (bad elements were [3, 7])")


  
    print("\n=== Fix #5: motion detection via redundant ray re-measurement ===")
    print("(first: naive random ray selection, for comparison)")
    no_motion = run_motion_check(motion_occurred=False)
    with_motion_random = run_motion_check(motion_occurred=True)
    print(f"no motion case (random rays):   {no_motion}")
    print(f"motion case (random rays):      {with_motion_random}")

  
    print("\n(now: targeted re-measurement of the rays crossing the scan's own flagged region)")
    n_elem_motion, coarse_size_motion = 16, 12
    rng_setup = np.random.default_rng(3000)
    labels_setup, _ = ham.build_phantom(rng_setup)
    positions_setup = srt.element_positions(n_elem_motion, radius_px=ham.GRID_SIZE / 2 - 3)
    A_setup, pairs_setup, skull_lengths_setup, interior_idx_setup, interior_mask_setup = \
        srt.build_ray_system_matrix_corrected(positions_setup, coarse_size_motion, ham.GRID_SIZE)
    raw_setup = srt.simulate_ray_measurements(labels_setup, positions_setup, pairs_setup, 1.0,
                                                snr_db=30, rng=rng_setup)
    skull_est_setup = ham.attenuation_db_per_cm("skull", 1.0)
    residual_setup = np.clip(raw_setup - skull_est_setup * skull_lengths_setup, 0, None)
    from sklearn.linear_model import Lasso
    lasso_setup = Lasso(alpha=0.001, positive=True, max_iter=10000)
    lasso_setup.fit(A_setup, residual_setup)
    recon_setup = np.zeros((coarse_size_motion, coarse_size_motion))
    for (r, c), idx in interior_idx_setup.items():
        recon_setup[r, c] = lasso_setup.coef_[idx]

  
    no_motion_targeted = run_motion_check(n_elements=n_elem_motion, motion_occurred=False, targeted=True,
                                            A=A_setup, interior_indices=interior_idx_setup,
                                            coarse_size=coarse_size_motion, recon=recon_setup, n_averages=32)
    with_motion_targeted = run_motion_check(n_elements=n_elem_motion, motion_occurred=True, targeted=True,
                                              A=A_setup, interior_indices=interior_idx_setup,
                                              coarse_size=coarse_size_motion, recon=recon_setup, n_averages=32)
    print(f"no motion case (targeted rays, 32x averaged): {no_motion_targeted}")
    print(f"motion case (targeted rays, 32x averaged):    {with_motion_targeted}")

    print("\n=== Fix #6: dosimetry screening vs FDA Track 3 limits (CALCULATED, NOT A CLEARANCE) ===")
    for voltage in [40, 60, 80, 100]:
        dosim = estimate_dosimetry(voltage, freq_mhz=1.0, burst_cycles=8, pulse_repetition_rate_hz=1000)
        print(f"  {voltage}Vpp: MI={dosim['mechanical_index']} (limit {dosim['mi_limit']}, "
              f"{'PASS' if dosim['mi_pass'] else 'FAIL'})  "
              f"Ispta={dosim['ispta_mw_cm2']}mW/cm2 (limit {dosim['ispta_limit_mw_cm2']}, "
              f"{'PASS' if dosim['ispta_pass'] else 'FAIL'})")

    print("\n=== Fix #7: repeat-and-average acquisition to improve accuracy ===")
    avg_results = run_averaging_comparison()


  
    import json
    with open("outputs/robustness_fixes_results.json", "w") as f:
        json.dump({
            "fix3_tof_windowing": tof_result,
            "fix4_coupling_check": {"results": coupling_results, "flagged": flagged,
                                      "expected_range": expected_range},
            "fix5_motion_detection": {
                "random_ray_selection": {"no_motion": no_motion, "with_motion": with_motion_random},
                "targeted_ray_selection": {"no_motion": no_motion_targeted, "with_motion": with_motion_targeted},
            },
            "fix6_dosimetry_screening": {
                str(v): estimate_dosimetry(v, 1.0, 8, 1000) for v in [40, 60, 80, 100]
            },
            "fix7_averaging_comparison": {str(k): v for k, v in avg_results.items()},
        }, f, indent=2, default=float)
    print("\nsaved robustness_fixes_results.json")
