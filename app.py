import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import head_acoustic_mapper as ham
import sparse_ray_tomography as srt




st.set_page_config(page_title="Headset Acoustic Mapper", layout="wide")
st.title("Headset Acoustic Mapper")
st.caption("Non-invasive acoustic plaque-mapping simulation and reconstruction pipeline")

tab1, tab2, tab3 = st.tabs(["Live Simulation", "Saved Validation Results", "Project Notes"])



with tab1:
    st.subheader("Run a phantom simulation live")
    col1, col2, col3 = st.columns(3)
    with col1:
        freq_mhz = st.select_slider("Drive frequency (MHz)", options=[0.5, 1.0, 2.0], value=1.0)
    with col2:
        n_angles = st.slider("Transducer angles", 30, 180, 90, step=10)
    with col3:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7)

    if st.button("Run simulation", type="primary"):
        with st.spinner("Building phantom and reconstructing..."):
            rng = np.random.default_rng(int(seed))
            labels, plaque_masks = ham.build_phantom(rng)
            atten_img = ham.labels_to_attenuation_image(labels, freq_mhz)
            sino, angles = ham.acquire_sinogram(atten_img, n_angles=n_angles)
            recon = ham.reconstruct(sino, angles)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            cmap_labels = np.zeros((ham.GRID_SIZE, ham.GRID_SIZE, 3))
            for name, t in ham.TISSUES.items():
                mask = labels == t["id"]
                rgb = np.array([int(t["color"][1:3], 16), int(t["color"][3:5], 16),
                                 int(t["color"][5:7], 16)]) / 255
                cmap_labels[mask] = rgb
            axes[0].imshow(cmap_labels)
            axes[0].set_title("Ground truth phantom")
            axes[0].axis("off")

            axes[1].imshow(sino, cmap="viridis", aspect="auto")
            axes[1].set_title("Sinogram (simulated headset signal)")

            im = axes[2].imshow(recon, cmap="inferno")
            axes[2].set_title(f"Reconstruction ({freq_mhz} MHz)")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046)

          
            st.pyplot(fig)
            st.info(f"Ground truth plaque clusters: {len(plaque_masks)} | "
                    f"Skull attenuation at {freq_mhz}MHz: "
                    f"{ham.attenuation_db_per_cm('skull', freq_mhz):.2f} dB/cm")

with tab2:
    st.subheader("Previously computed validation results")
    st.caption("These come from the full multi-phantom / noise / calibration test suites "
               "(see the .json files in this repo) -- too slow to rerun live on every page load.")

    result_files = {
        "Multi phantom validation (20 trials, clean + noise)": "outputs/multi_phantom_validation.json",
        "Sparse ray tomography ": "sparse_ray_tomography_results.json",
        "Bench calibration results": "calibration_results.json",
        "Integrated calibration -> reconstruction pipeline": "integrated_pipeline_results.json",
        "Robustness fixes (timing, coupling, motion, safety, averaging)": "robustness_fixes_results.json",
    }

  
    for label, path in result_files.items():
        if os.path.exists(path):
            with st.expander(label):
                with open(path) as f:
                    st.json(json.load(f))
        else:
            st.warning(f"{label}: file not found at `{path}` -- "
                       f"make sure it's committed to the repo alongside app.py")

with tab3:
    st.subheader("Honest project status")
    st.markdown("""
    - The reconstruction algorithm & bench calibration procedure are real
      & internally consistent & would run unmodified on real hardware
      captures in this data format.
    - **The  biggest unaddressed gap is physics here and surprisingly not software**: this
      pipeline assumes straight ray sound propagation.
    - Absolute reconstruction accuracy is modest (Dice ~0.1-0.42
      across conditions. Even in idealized simulation) this is a working
      research signal and not a reliable measurement tool yet.
    - Safety figures shown elsewhere in this repo are a **calculated
      screening check only**. Pls NOTE that it does not have a regulatory clearance.


    """)
