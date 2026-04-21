def main():

    import sys
    sys.path.insert(0, '/mnt/nfs/source/')

    from ase.build import molecule
    import numpy as np

    from descriptor_tools import (
        SingleCenterDescriptor,
        descriptor_catalog,
        sensitivity_eigendecomposition,
        plot_sensitivity_spectra_along_manifold,
        plot_sensitivity_spectra,
    )

    # -----------------------
    # Settings
    # -----------------------
    atoms = molecule("NH3")
    #atoms.rattle(stdev=0.03, seed=1)
    atoms.center(vacuum=8.0)
    atoms_name = "NH3" #atoms.get_chemical_formula()
    descriptor_mode = "local"   # "local" or "global"
    center_index = 0
    n_procs = 8
    eps = 1e-4

    calc_cfg = dict(
        mace_model_paths="/mnt/nfs/spd_paper/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/ml_training/global_training_1/mace_global256_CoMoFeNiCuNH.model",
        mace_device="cpu",
        mace_default_dtype="float64",
        mace_enable_cueq=True,
        mace_invariants_only=False,
        om_rcut=3.0,
        soap_rcut=6.0,
        soap_n_max=12,
        soap_l_max=12,
        soap_sigma=0.5,
        acsf_rcut=6.0,
    )

    descs = descriptor_catalog(mode=descriptor_mode, **calc_cfg)

    if descriptor_mode == "local":
        descriptor_fns = {
            "OMDescriptor": SingleCenterDescriptor(descs["OMDescriptor"], center_index=center_index),
            "SOAPDescriptor": SingleCenterDescriptor(descs["SOAPDescriptor"], center_index=center_index),
            "ACSFDescriptor": SingleCenterDescriptor(descs["ACSFDescriptor"], center_index=center_index),
            "MACEDescriptor": SingleCenterDescriptor(descs["MACEDescriptor"], center_index=center_index),
        }
        ref_fn = descriptor_fns["OMDescriptor"]
    else:
        descriptor_fns = {
            "OMDescriptor": descs["OMDescriptor"],
            "SOAPDescriptor": descs["SOAPDescriptor"],
            "ACSFDescriptor": descs["ACSFDescriptor"],
            "MACEDescriptor": descs["MACEDescriptor"],
        }
        ref_fn = descriptor_fns["OMDescriptor"]

    """
    # Trace manifold using OM as the reference descriptor
    manifold_path, eig_history = trace_quasi_constant_manifold(
        atoms=atoms,
        descriptor_fn=ref_fn,
        active_indices=np.arange(len(atoms), dtype=int),
        n_steps=n_steps,
        step_size=step_size,
        eps=eps,
        n_procs=n_procs,
    )
    plot_sensitivity_spectra_along_manifold(
        manifold_path=manifold_path,
        descriptor_fns=descriptor_fns,
        active_indices=np.arange(len(atoms), dtype=int),
        eps=eps,
        n_procs=n_procs,
        n_show=10,
        skip_rigid=6,
        title=f"Sensitivity spectra along manifold for {atoms_name} ({descriptor_mode})",
        save_path=f"sensitivity_spectra_along_manifold_{atoms_name}_{descriptor_mode}.png",
    )
    """

    plot_sensitivity_spectra(
        atoms=atoms,
        descriptor_fns=descriptor_fns,
        active_indices=np.arange(len(atoms), dtype=int),
        eps=eps,
        n_procs=n_procs,
        normalize=True,
        title=f"Sensitivity spectra for {atoms_name} ({descriptor_mode})",
        save_path=f"sensitivity_spectra_{atoms_name}_{descriptor_mode}.png",
        show_indices=False,
        save_interactive_path=None,
    )

if __name__ == "__main__":
    main()
