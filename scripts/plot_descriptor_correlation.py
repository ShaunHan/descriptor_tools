def main():

    import sys
    sys.path.insert(0, '/mnt/nfs/source/')
    from ase.io import read
    import numpy as np

    from descriptor_tools import (
        descriptor_catalog,
        plot_local_pairwise_correlation,
        plot_global_pairwise_correlation,
    )

    xyz_file = "/mnt/nfs/ml_training/C60/data/C60_aims_all.xyz"
    descriptor_mode = "local"  # "local" or "global"
    n_atoms_per_config = 60   # used only for local
    n_procs = 8
    seed = 0
    desc1 = 'OM'
    desc2 = 'ACSF'

    calc_cfg = dict(
        mace_model_paths="mace_global256_C60.model",
        mace_device="cpu",
        mace_default_dtype="float64",
        mace_enable_cueq=False,
        mace_invariants_only=False,
        om_rcut=3.0,
        soap_rcut=6.0,
        soap_n_max=12,
        soap_l_max=12,
        soap_sigma=0.5,
        acsf_rcut=6.0,
    )

    atoms_list = read(xyz_file, index=":1")
    descs = descriptor_catalog(mode=descriptor_mode, **calc_cfg)

    if descriptor_mode == "local":
        plot_local_pairwise_correlation(
            atoms_list=atoms_list,
            descriptor_fn_A=descs[f"{desc1.upper()}Descriptor"],
            descriptor_fn_B=descs[f"{desc2.upper()}Descriptor"],
            n_atoms_per_config=n_atoms_per_config,
            n_procs=n_procs,
            normalize=True,
            title=f"Local {desc1.upper()} vs local {desc2.upper()} correlation",
            save_path=f"local_{desc1.lower()}_vs_{desc2.lower()}_correlation.png",
            seed=seed,
            show_indices=True,
            save_interactive_path=f"local_{desc1.lower()}_vs_{desc2.lower()}_correlation.pkl",
        )
    else:
        plot_global_pairwise_correlation(
            atoms_list=atoms_list,
            descriptor_fn_A=descs[f"{desc1.upper()}Descriptor"],
            descriptor_fn_B=descs[f"{desc2.upper()}Descriptor"],
            n_procs=n_procs,
            normalize=True,
            title=f"Global {desc1.upper()} vs global {desc2.upper()} correlation",
            save_path=f"global_{desc1.lower()}_vs_{desc2.lower()}_correlation.png",
            seed=seed,
            show_indices=True,
            save_interactive_path=f"global_{desc1.lower()}_vs_{desc2.lower()}_correlation.pkl",
        )


if __name__ == "__main__":
    main()
