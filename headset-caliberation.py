import numpy as np
import sys
import head_acoustic_mapper as ham




def simulate_bench_capture(true_path_length_cm, drive_voltage_v, gain_db,
                            true_attenuation_db_per_cm, adc_full_scale=4095,
                            rng=None):
  
    if rng is None:
        rng = np.random.default_rng(0)

    tx_amplitude = drive_voltage_v * 8.0 
    spreading_loss = 1.0 / max(true_path_length_cm, 0.1)
    atten_loss = 10 ** (-true_attenuation_db_per_cm * true_path_length_cm / 20.0)
    gain_linear = 10 ** (gain_db / 20.0)

    signal_amplitude = tx_amplitude * spreading_loss * atten_loss * gain_linear
    electronic_noise_rms = 3.0  
    noise = rng.normal(0, electronic_noise_rms)

    raw = signal_amplitude + noise
    clipped = np.clip(raw, 0, adc_full_scale)
    saturated = raw > adc_full_scale
    return clipped, saturated, electronic_noise_rms


def run_gain_voltage_sweep(reference_path_length_cm=4.0, true_attenuation=0.58,
                            voltages=(20, 40, 60, 80, 100), gains_db=(0, 10, 20, 30, 40),
                            n_repeats=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    print(f"{'V(pp)':<8}{'gain(dB)':<10}{'mean_amp':<12}{'noise_rms':<12}{'SNR(dB)':<10}{'saturated':<10}")
    results = []
    for v in voltages:
        for g in gains_db:
            captures = []
            sat_count = 0
            for _ in range(n_repeats):
                cap, sat, noise_rms = simulate_bench_capture(
                    reference_path_length_cm, v, g, true_attenuation, rng=rng)
                captures.append(cap)
                sat_count += sat
            mean_amp = np.mean(captures)
            snr_db = 20 * np.log10(mean_amp / noise_rms) if mean_amp > 0 else -999
            saturated = sat_count > 0
            results.append(dict(voltage=v, gain_db=g, mean_amplitude=mean_amp,
                                 noise_rms=noise_rms, snr_db=snr_db, saturated=saturated))
            print(f"{v:<8}{g:<10}{mean_amp:<12.1f}{noise_rms:<12.2f}{snr_db:<10.1f}{str(saturated):<10}")
    return results


def select_calibrated_settings(sweep_results, adc_full_scale=4095, headroom_frac=0.85):
    # Made sure to pick the highest-SNR setting that stays comfortably below ADC full scale
    safe = [r for r in sweep_results if r["mean_amplitude"] < adc_full_scale * headroom_frac
            and not r["saturated"]]
    if not safe:
        raise RuntimeError("no non-saturating setting found in sweep range -- widen the sweep")
    best = max(safe, key=lambda r: r["snr_db"])
    return best




def fit_transmission_loss_curve(path_lengths_cm, measured_amplitudes, drive_voltage,
                                 gain_db):
.
    path_lengths_cm = np.asarray(path_lengths_cm)
    measured_amplitudes = np.asarray(measured_amplitudes)
    y = np.log(measured_amplitudes * path_lengths_cm)
    A = np.vstack([path_lengths_cm, np.ones_like(path_lengths_cm)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    alpha_nepers_per_cm = -slope
    alpha_db_per_cm = alpha_nepers_per_cm * 20 / np.log(10)
    system_gain_constant = np.exp(intercept)
    return alpha_db_per_cm, system_gain_constant


def find_safe_calibration_gain(voltage, lengths_cm, true_attenuation, adc_full_scale=4095,
                                min_amplitude=15.0, max_amplitude_frac=0.85, rng=None):
   
    if rng is None:
        rng = np.random.default_rng(1)

    for gain_db in range(0, 61, 2):
        ok = True
        for L in lengths_cm:
            cap, sat, noise_rms = simulate_bench_capture(L, voltage, gain_db, true_attenuation, rng=rng)
            if sat or cap < min_amplitude or cap > adc_full_scale * max_amplitude_frac:
                ok = False
                break
        if ok:
            return gain_db
    return None


def calibrate_skull_coefficient(freq_mhz, drive_voltage, gain_db, n_reference_lengths=8,
                                 length_range_cm=(2.0, 8.0), n_repeats=5, rng=None,
                                 auto_gain=True):
    if rng is None:
        rng = np.random.default_rng(7)

    true_skull_coef = ham.attenuation_db_per_cm("skull", freq_mhz)
    lengths = np.linspace(*length_range_cm, n_reference_lengths)

    if auto_gain:
        safe_gain = find_safe_calibration_gain(drive_voltage, lengths, true_skull_coef, rng=rng)
        if safe_gain is None:
            return dict(error=(
                "no single fixed gain keeps all reference distances in a valid "
                "dynamic range. This distance needs time gain compensation "
                "Narrow the calibration distance range / implement TGC."
            ))
        if safe_gain != gain_db:
            print(f"  (note: single-distance SNR-optimal gain {gain_db}dB clips/underflows "
                  f"across the {length_range_cm} cm calibration range; using {safe_gain}dB "
                  f"for this sweep instead ; a real TGC implementation uses both)")
        gain_db = safe_gain

    all_estimates = []
    for _ in range(n_repeats):
        amplitudes = []
        for L in lengths:
            cap, sat, _ = simulate_bench_capture(L, drive_voltage, gain_db, true_skull_coef, rng=rng)
            amplitudes.append(max(cap, 1e-6))
        alpha_est, _ = fit_transmission_loss_curve(lengths, amplitudes, drive_voltage, gain_db)
        all_estimates.append(alpha_est)

    mean_est = np.mean(all_estimates)
    std_est = np.std(all_estimates)
    return dict(
        estimated_skull_coef_db_cm=mean_est,
        uncertainty_std=std_est,
        relative_error_pct=100 * abs(mean_est - true_skull_coef) / true_skull_coef,
        n_reference_measurements=n_reference_lengths * n_repeats,
        calibration_gain_used_db=gain_db,
    )


if __name__ == "__main__":
    print("=== Step 1: gain/voltage sweep against a fixed reference path ===")
    sweep = run_gain_voltage_sweep()
    best = select_calibrated_settings(sweep)
    print(f"\nSelected setting: {best['voltage']}Vpp drive, {best['gain_db']}dB gain "
          f"-> SNR {best['snr_db']:.1f}dB, mean amplitude {best['mean_amplitude']:.0f} "
          f"(headroom below {int(4095*0.85)})")

    print("\n=== Step 2: skull-coefficient calibration from reference-phantom measurements ===")
    true_val = ham.attenuation_db_per_cm("skull", 1.0)
    print(f"true (simulated) skull coefficient: {true_val:.3f} dB/cm")


    calib = calibrate_skull_coefficient(freq_mhz=1.0, drive_voltage=best["voltage"],
                                          gain_db=best["gain_db"], length_range_cm=(1.0, 3.0))
    if "error" in calib:
        print(f"CALIBRATION FAILED: {calib['error']}")
        calib = None
    else:
        print(f"calibrated estimate:                {calib['estimated_skull_coef_db_cm']:.3f} "
              f"+/- {calib['uncertainty_std']:.3f} dB/cm "
              f"(using {calib['calibration_gain_used_db']}dB gain, "
              f"vs {best['gain_db']}dB from step 1)")
        print(f"relative error:                     {calib['relative_error_pct']:.1f}%")
        print(f"(from {calib['n_reference_measurements']} reference-phantom measurements)")

    print("\n=== Same calibration attempted at the original wider (2,8)cm span, for comparison ===")
    calib_wide = calibrate_skull_coefficient(freq_mhz=1.0, drive_voltage=best["voltage"],
                                               gain_db=best["gain_db"], length_range_cm=(2.0, 8.0))
    if "error" in calib_wide:
        print(f"CALIBRATION FAILED (as expected): {calib_wide['error']}")
    else:
        print(f"unexpectedly succeeded: {calib_wide}")


  
    import json
    with open("outputs/calibration_results.json", "w") as f:
        json.dump({
            "selected_drive_settings": best,
            "skull_coefficient_calibration_realistic_range_1_3cm": calib,
            "skull_coefficient_calibration_wide_range_2_8cm": calib_wide,
        }, f, indent=2, default=float)
    print("\nsaved calibration_results.json")
