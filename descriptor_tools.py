"""
Reusable descriptor wrappers and plotting utilities for OM / SOAP / ACSF / MACE.

Features
--------
- local and global descriptors
- multiprocessing descriptor evaluation (serial only when n_procs == 1)
- finite-difference sensitivity matrices
- quasi-constant manifold tracing
- manifold visualization
- local/global pairwise correlation plots
- sensitivity eigenvalue plots
- optional interactive hover labels and pickleable figure saving

Notes
-----
For multiprocessing with the 'spawn' start method, the calling script must still
use the standard main guard::

    if __name__ == '__main__':
        main()
"""

from __future__ import annotations

import math
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from matplotlib.colors import LogNorm
from scipy.linalg import eigh
from ase import Atoms
from dscribe.descriptors import ACSF, SOAP
from minimahopping.omfp.OverlapMatrixFingerprint import OverlapMatrixFingerprint
from minimahopping.mh import periodictable


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _to_1d_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _pad_or_truncate(fp, target_len: int) -> np.ndarray:
    fp = _to_1d_array(fp)
    if fp.size == target_len:
        return fp
    out = np.zeros(int(target_len), dtype=float)
    n = min(fp.size, target_len)
    out[:n] = fp[:n]
    return out


def _descriptor_species(atoms: Atoms) -> List[str]:
    species = []
    seen = set()
    for s in atoms.get_chemical_symbols():
        if s not in seen:
            seen.add(s)
            species.append(s)
    return species


def _infer_centers(atoms: Atoms, centers: Optional[Sequence[int]]) -> np.ndarray:
    if centers is None:
        return np.arange(len(atoms), dtype=int)
    return np.asarray(centers, dtype=int)


def _as_list_of_vectors(out) -> List[np.ndarray]:
    """Normalize descriptor outputs to a list of 1D vectors."""
    if isinstance(out, list):
        return [_to_1d_array(v) for v in out]

    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        return [arr.reshape(-1)]
    if arr.ndim == 2:
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
        return [arr[i].reshape(-1) for i in range(arr.shape[0])]
    return [_to_1d_array(arr)]


def _labels_to_text(label) -> str:
    return str(label)


# -----------------------------------------------------------------------------
# Interactive hover support
# -----------------------------------------------------------------------------

class InteractiveFigure:
    def __init__(self, fig, hover_artists=None):
        self.fig = fig
        self.hover_artists = hover_artists or []

    def __getstate__(self):
        return {
            "fig": self.fig,
            "hover_artists": self.hover_artists,
        }

    def __setstate__(self, state):
        self.fig = state["fig"]
        self.hover_artists = state.get("hover_artists", [])

    def _attach_hover(self):
        self._cursors = []
        for artist in self.hover_artists:
            labels = getattr(artist, "_descriptor_hover_labels", None)
            if labels is None:
                continue
            cursor = mplcursors.cursor(artist, hover=True)
            @cursor.connect("add")
            def _on_add(sel, labels=labels):
                sel.annotation.set_text(str(labels[sel.index]))
            self._cursors.append(cursor)

    def show(self, *args, **kwargs):
        self._attach_hover()
        return self.fig.show(*args, **kwargs)

    def save_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getattr__(self, name):
        return getattr(self.fig, name)


def _make_hover_scatter(ax, x, y, labels):
    # invisible overlay; does not change the plot appearance
    sc = ax.scatter(
        x, y,
        s=20,
        alpha=0.0,
        linewidths=0.0,
        edgecolors="none",
        picker=True,
    )
    sc._descriptor_hover_labels = labels
    return sc


# -----------------------------------------------------------------------------
# Descriptor wrappers
# -----------------------------------------------------------------------------

class OMDescriptor:
    """
    Overlap Matrix fingerprint.

    mode='local'  -> returns local fingerprints for selected centers
    mode='global' -> returns one global fingerprint
    """

    def __init__(self, mode: str = "global", rcut: float = 3.0):
        self.mode = mode.lower()
        self.rcut = float(rcut)

    def __call__(self, atoms: Atoms, centers: Optional[Sequence[int]] = None):
        lmn = {}
        for Z in np.unique(atoms.get_atomic_numbers()):
            rc = periodictable.getRcov_n(int(Z))
            lmn[int(Z)] = [(rc, "s"), (rc, "p")]

        omfp = OverlapMatrixFingerprint(lmn, rcut=self.rcut)
        ats = atoms.get_positions()
        els = atoms.get_atomic_numbers()

        if self.mode == "global":
            return _to_1d_array(omfp.globalFingerprint(ats, els))

        local_fps = omfp.fingerprint(
            ats,
            els,
            lat=atoms.get_cell().array if atoms.pbc.any() else None,
        )
        local_fps = _as_list_of_vectors(local_fps)
        idx = _infer_centers(atoms, centers)
        return [local_fps[int(i)] for i in idx]


class SOAPDescriptor:
    """
    SOAP fingerprint.

    mode='local'  -> local SOAP at selected centers
    mode='global' -> global averaged SOAP
    """

    def __init__(
        self,
        mode: str = "global",
        rcut: float = 6.0,
        n_max: int = 12,
        l_max: int = 12,
        sigma: float = 0.5,
    ):
        self.mode = mode.lower()
        self.rcut = float(rcut)
        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.sigma = float(sigma)

    def __call__(self, atoms: Atoms, centers: Optional[Sequence[int]] = None):
        soap = SOAP(
            species=_descriptor_species(atoms),
            periodic=bool(True in atoms.pbc),
            r_cut=self.rcut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            average="inner" if self.mode == "global" else "off",
            sparse=False,
        )
        if self.mode == "global":
            return _to_1d_array(soap.create(atoms))

        idx = _infer_centers(atoms, centers)
        fp = np.asarray(soap.create(atoms, centers=list(map(int, idx))), dtype=float)
        return [_to_1d_array(fp[i]) for i in range(fp.shape[0])]


class ACSFDescriptor:
    """
    ACSF fingerprint.

    mode='local'  -> local ACSF at selected centers
    mode='global' -> averaged over all atomic centers
    """

    def __init__(
        self,
        mode: str = "global",
        rcut: float = 6.0,
        g2_params: Optional[Sequence[Sequence[float]]] = None,
        g4_params: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.mode = mode.lower()
        self.rcut = float(rcut)
        self.g2_params = g2_params if g2_params is not None else [[1, 1], [1, 2], [1, 3]]
        self.g4_params = g4_params if g4_params is not None else [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]

    def __call__(self, atoms: Atoms, centers: Optional[Sequence[int]] = None):
        acsf = ACSF(
            species=_descriptor_species(atoms),
            periodic=bool(True in atoms.pbc),
            r_cut=self.rcut,
            g2_params=self.g2_params,
            g4_params=self.g4_params,
            sparse=False,
        )

        if self.mode == "global":
            fp = np.asarray(acsf.create(atoms), dtype=float)

            # DScribe may return shape (n_atoms, n_features) or (n_features,)
            if fp.ndim == 1:
                return _to_1d_array(fp)

            return _to_1d_array(np.mean(fp, axis=0))

        idx = _infer_centers(atoms, centers)
        fp = np.asarray(acsf.create(atoms, centers=list(map(int, idx))), dtype=float)
        return [_to_1d_array(fp[i]) for i in range(fp.shape[0])]


class MACEDescriptor:
    """
    MACE descriptor wrapper.

    mode='local'  -> local node features
    mode='global' -> permutation-invariant pooled global descriptor
    """

    def __init__(
        self,
        model_paths,
        mode: str = "global",
        device: str = "cuda",
        default_dtype: str = "float64",
        enable_cueq: bool = True,
        invariants_only: bool = False,
    ):
        self.model_paths = model_paths
        self.mode = mode.lower()
        self.device = device
        self.default_dtype = default_dtype
        self.enable_cueq = enable_cueq
        self.invariants_only = invariants_only
        self._calc = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_calc"] = None
        return state

    def _get_calc(self):
        if self._calc is None:
            from mace.calculators.mace import MACECalculator
            self._calc = MACECalculator(
                model_paths=self.model_paths,
                device=self.device,
                default_dtype=self.default_dtype,
                enable_cueq=self.enable_cueq,
            )
        return self._calc

    def __call__(self, atoms: Atoms, centers: Optional[Sequence[int]] = None):
        calc = self._get_calc()
        desc = calc.get_descriptors(
            atoms,
            invariants_only=self.invariants_only,
            global_descriptor=(self.mode == "global"),
        )

        if isinstance(desc, list):
            desc = desc[0]

        arr = np.asarray(desc, dtype=float)

        if self.mode == "global":
            return arr.reshape(-1)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        idx = _infer_centers(atoms, centers)
        return [arr[int(i)].reshape(-1) for i in idx]


class SingleCenterDescriptor:
    """Turn any local descriptor into a single-center callable returning one vector."""

    def __init__(self, descriptor, center_index: int = 0):
        self.descriptor = descriptor
        self.center_index = int(center_index)

    def __getstate__(self):
        return self.__dict__.copy()

    def __call__(self, atoms: Atoms):
        out = self.descriptor(atoms, centers=[self.center_index])
        if isinstance(out, list):
            return _to_1d_array(out[0])
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1)
        return arr[0].reshape(-1)


def descriptor_catalog(
    mode: str = "global",
    mace_model_paths=None,
    mace_device: str = "cuda",
    mace_default_dtype: str = "float64",
    mace_enable_cueq: bool = True,
    mace_invariants_only: bool = False,
    om_rcut: float = 3.0,
    soap_rcut: float = 6.0,
    soap_n_max: int = 12,
    soap_l_max: int = 12,
    soap_sigma: float = 0.5,
    acsf_rcut: float = 6.0,
):
    return {
        "OMDescriptor": OMDescriptor(mode=mode, rcut=om_rcut),
        "SOAPDescriptor": SOAPDescriptor(
            mode=mode, rcut=soap_rcut, n_max=soap_n_max, l_max=soap_l_max, sigma=soap_sigma
        ),
        "ACSFDescriptor": ACSFDescriptor(mode=mode, rcut=acsf_rcut),
        "MACEDescriptor": MACEDescriptor(
            model_paths=mace_model_paths,
            mode=mode,
            device=mace_device,
            default_dtype=mace_default_dtype,
            enable_cueq=mace_enable_cueq,
            invariants_only=mace_invariants_only,
        ),
    }


# -----------------------------------------------------------------------------
# Descriptor evaluation helpers
# -----------------------------------------------------------------------------

def _sample_centers_per_config(
    atoms_list: Sequence[Atoms],
    n_atoms_per_config: Optional[int] = None,
    seed: int = 0,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    centers_list = []
    for atoms in atoms_list:
        n_atoms = len(atoms)
        if n_atoms_per_config is None or int(n_atoms_per_config) >= n_atoms:
            centers = np.arange(n_atoms, dtype=int)
        else:
            centers = np.sort(rng.choice(n_atoms, size=int(n_atoms_per_config), replace=False))
        centers_list.append(centers)
    return centers_list


_GLOBAL_ATOMS_LIST = None
_GLOBAL_DESCRIPTOR_FN = None


def _global_worker_init(atoms_list, descriptor_fn):
    global _GLOBAL_ATOMS_LIST, _GLOBAL_DESCRIPTOR_FN
    _GLOBAL_ATOMS_LIST = atoms_list
    _GLOBAL_DESCRIPTOR_FN = descriptor_fn


def _global_worker(i):
    return i, _to_1d_array(_GLOBAL_DESCRIPTOR_FN(_GLOBAL_ATOMS_LIST[i]))


def compute_global_descriptors(
    atoms_list: Sequence[Atoms],
    descriptor_fn,
    n_procs: int = 1,
):
    if n_procs == 1:
        return [_to_1d_array(descriptor_fn(a)) for a in atoms_list]

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(n_procs),
        initializer=_global_worker_init,
        initargs=(atoms_list, descriptor_fn),
    ) as pool:
        results = pool.map(_global_worker, range(len(atoms_list)))

    results.sort(key=lambda x: x[0])
    return [fp for _, fp in results]


_LOCAL_ATOMS_LIST = None
_LOCAL_DESCRIPTOR_FN = None
_LOCAL_CENTERS_LIST = None


def _local_worker_init(atoms_list, descriptor_fn, centers_list):
    global _LOCAL_ATOMS_LIST, _LOCAL_DESCRIPTOR_FN, _LOCAL_CENTERS_LIST
    _LOCAL_ATOMS_LIST = atoms_list
    _LOCAL_DESCRIPTOR_FN = descriptor_fn
    _LOCAL_CENTERS_LIST = centers_list


def _local_worker(i):
    atoms = _LOCAL_ATOMS_LIST[i]
    centers = _LOCAL_CENTERS_LIST[i]
    fps = _local_descriptor_as_list(_LOCAL_DESCRIPTOR_FN, atoms, centers)
    return i, fps


def _local_descriptor_as_list(descriptor_fn, atoms: Atoms, centers: np.ndarray) -> List[np.ndarray]:
    out = descriptor_fn(atoms, centers=centers)
    return _as_list_of_vectors(out)


def compute_local_descriptors(
    atoms_list: Sequence[Atoms],
    descriptor_fn,
    centers_list: Optional[Sequence[np.ndarray]] = None,
    n_atoms_per_config: Optional[int] = None,
    n_procs: int = 1,
    seed: int = 0,
):
    if centers_list is None:
        centers_list = _sample_centers_per_config(
            atoms_list, n_atoms_per_config=n_atoms_per_config, seed=seed
        )

    if n_procs == 1:
        all_fps = []
        for atoms, centers in zip(atoms_list, centers_list):
            all_fps.extend(_local_descriptor_as_list(descriptor_fn, atoms, centers))
        return all_fps, centers_list

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(n_procs),
        initializer=_local_worker_init,
        initargs=(atoms_list, descriptor_fn, centers_list),
    ) as pool:
        results = pool.map(_local_worker, range(len(atoms_list)))

    results.sort(key=lambda x: x[0])
    all_fps = []
    for _, fps in results:
        all_fps.extend(fps)
    return all_fps, centers_list


# -----------------------------------------------------------------------------
# Noise scales
# -----------------------------------------------------------------------------

def _compute_noise_scale_global(
    atoms_list: Sequence[Atoms],
    descriptor_fn,
    target_len: int,
    perturb: float = 0.02,
    n_samples: int = 200,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_take = min(int(n_samples), len(atoms_list))
    idx = rng.choice(len(atoms_list), size=n_take, replace=False)

    dists = []
    for i in idx:
        atoms = atoms_list[i].copy()
        R0 = atoms.get_positions()
        delta = rng.uniform(-perturb, perturb, size=R0.shape)
        fp0 = _pad_or_truncate(descriptor_fn(atoms), target_len)

        atoms.set_positions(R0 + delta)
        fp1 = _pad_or_truncate(descriptor_fn(atoms), target_len)
        dists.append(np.linalg.norm(fp1 - fp0))

    return float(np.mean(dists)) if len(dists) > 0 else 1.0


def _compute_noise_scale_local(
    atoms_list: Sequence[Atoms],
    descriptor_fn,
    centers_list: Sequence[np.ndarray],
    target_len: int,
    perturb: float = 0.02,
    n_samples: int = 200,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_take = min(int(n_samples), len(atoms_list))
    idx = rng.choice(len(atoms_list), size=n_take, replace=False)

    dists = []
    for i in idx:
        atoms = atoms_list[i].copy()
        centers = centers_list[i]
        R0 = atoms.get_positions()

        fp0_list = [
            _pad_or_truncate(fp, target_len)
            for fp in _local_descriptor_as_list(descriptor_fn, atoms, centers)
        ]

        delta = rng.uniform(-perturb, perturb, size=R0.shape)
        atoms.set_positions(R0 + delta)

        fp1_list = [
            _pad_or_truncate(fp, target_len)
            for fp in _local_descriptor_as_list(descriptor_fn, atoms, centers)
        ]

        for a, b in zip(fp0_list, fp1_list):
            dists.append(np.linalg.norm(b - a))

    return float(np.mean(dists)) if len(dists) > 0 else 1.0


# -----------------------------------------------------------------------------
# Sensitivity matrix tools
# -----------------------------------------------------------------------------

_FD_GLOBAL_ATOMS = None
_FD_GLOBAL_DESCRIPTOR_FN = None
_FD_GLOBAL_EPS = None
_FD_GLOBAL_R0 = None


def _flat_descriptor(descriptor_fn, atoms: Atoms) -> np.ndarray:
    return _to_1d_array(descriptor_fn(atoms))


def _fd_worker_init(atoms, descriptor_fn, eps, r0):
    global _FD_GLOBAL_ATOMS, _FD_GLOBAL_DESCRIPTOR_FN, _FD_GLOBAL_EPS, _FD_GLOBAL_R0
    _FD_GLOBAL_ATOMS = atoms
    _FD_GLOBAL_DESCRIPTOR_FN = descriptor_fn
    _FD_GLOBAL_EPS = eps
    _FD_GLOBAL_R0 = r0


def _fd_worker(task):
    a, c, sign = task
    atoms = _FD_GLOBAL_ATOMS.copy()
    R = _FD_GLOBAL_R0.copy()
    R[a, c] += sign * _FD_GLOBAL_EPS
    atoms.set_positions(R)
    fp = _flat_descriptor(_FD_GLOBAL_DESCRIPTOR_FN, atoms)
    return a, c, sign, fp


def finite_difference_jacobian(
    atoms: Atoms,
    descriptor_fn,
    active_indices: Optional[Sequence[int]] = None,
    eps: float = 1e-4,
    n_procs: int = 1,
):
    atoms = atoms.copy()
    R0 = atoms.get_positions()
    n_atoms = len(atoms)

    if active_indices is None:
        active_indices = np.arange(n_atoms, dtype=int)
    else:
        active_indices = np.asarray(active_indices, dtype=int)

    F0 = _flat_descriptor(descriptor_fn, atoms)
    n_feat = F0.size
    J = np.zeros((n_feat, 3 * len(active_indices)), dtype=float)

    if n_procs == 1:
        col = 0
        for a in active_indices:
            for c in range(3):
                dR = np.zeros_like(R0)
                dR[a, c] = eps

                atoms.set_positions(R0 + dR)
                F_plus = _flat_descriptor(descriptor_fn, atoms)

                atoms.set_positions(R0 - dR)
                F_minus = _flat_descriptor(descriptor_fn, atoms)

                J[:, col] = (F_plus - F_minus) / (2.0 * eps)
                col += 1

        atoms.set_positions(R0)
        return F0, J

    global _FD_GLOBAL_ATOMS, _FD_GLOBAL_DESCRIPTOR_FN, _FD_GLOBAL_EPS, _FD_GLOBAL_R0
    _FD_GLOBAL_ATOMS = atoms.copy()
    _FD_GLOBAL_DESCRIPTOR_FN = descriptor_fn
    _FD_GLOBAL_EPS = float(eps)
    _FD_GLOBAL_R0 = R0.copy()

    tasks = []
    for a in active_indices:
        for c in range(3):
            tasks.append((int(a), int(c), +1))
            tasks.append((int(a), int(c), -1))

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(n_procs),
        initializer=_fd_worker_init,
        initargs=(_FD_GLOBAL_ATOMS, _FD_GLOBAL_DESCRIPTOR_FN, _FD_GLOBAL_EPS, _FD_GLOBAL_R0),
    ) as pool:
        results = pool.map(_fd_worker, tasks)

        result_map = {}
        for a, c, sign, fp in results:
            result_map[(a, c, sign)] = fp

    col = 0
    for a in active_indices:
        for c in range(3):
            F_plus = result_map[(int(a), int(c), +1)]
            F_minus = result_map[(int(a), int(c), -1)]
            J[:, col] = (F_plus - F_minus) / (2.0 * eps)
            col += 1

    atoms.set_positions(R0)
    return F0, J


def sensitivity_eigendecomposition(
    atoms: Atoms,
    descriptor_fn,
    active_indices: Optional[Sequence[int]] = None,
    descending: bool = False,
    eps: float = 1e-4,
    n_procs: int = 1,
    normalize: bool = True,
):
    _, J = finite_difference_jacobian(
        atoms=atoms,
        descriptor_fn=descriptor_fn,
        active_indices=active_indices,
        eps=eps,
        n_procs=n_procs,
    )
    S = J.T @ J
    evals, evecs = eigh(S)
    if descending:
        order = np.argsort(evals)[::-1]
    else:
        order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    if normalize:
        max_eval = evals[0] if descending else evals[-1]
        if max_eval > 0:
            evals = evals / max_eval
    return evals, evecs, S, J


def mode_vector_to_displacement(
    atoms: Atoms,
    mode_vec: np.ndarray,
    active_indices: Optional[Sequence[int]] = None,
    normalize_to: float = 0.2,
):
    n_atoms = len(atoms)
    if active_indices is None:
        active_indices = np.arange(n_atoms, dtype=int)
    else:
        active_indices = np.asarray(active_indices, dtype=int)

    disp = np.zeros((n_atoms, 3), dtype=float)
    mv = np.asarray(mode_vec, dtype=float).reshape(len(active_indices), 3)
    disp[active_indices] = mv

    norms = np.linalg.norm(disp[active_indices], axis=1)
    max_norm = np.max(norms) if len(norms) > 0 else 1.0
    if max_norm > 0:
        disp *= (normalize_to / max_norm)

    return disp


def _first_nonrigid_mode_index(evals, rigid_tol=None):
    evals = np.asarray(evals, dtype=float)
    if rigid_tol is None:
        rigid_tol = max(1e-12, 1e-8 * float(np.max(evals) if evals.size else 1.0))
    idx = np.where(evals > rigid_tol)[0]
    if len(idx) == 0:
        raise RuntimeError("No non-rigid mode found above tolerance.")
    return int(idx[0])


def trace_quasi_constant_manifold(
    atoms: Atoms,
    descriptor_fn,
    active_indices: Optional[Sequence[int]] = None,
    n_steps: int = 30,
    step_size: float = 0.03,
    descending: bool = False,
    eps: float = 1e-4,
    n_procs: int = 1,
    rigid_tol=None,
):
    current = atoms.copy()
    manifold_path = [current.copy()]
    eig_history = []

    for _ in range(int(n_steps)):
        evals, evecs, _, _ = sensitivity_eigendecomposition(
            atoms=current,
            descriptor_fn=descriptor_fn,
            active_indices=active_indices,
            descending=descending,
            eps=eps,
            n_procs=n_procs,
            normalize=True,
        )
        eig_history.append(evals)

        idx = _first_nonrigid_mode_index(evals, rigid_tol=rigid_tol)
        mode_vec = evecs[:, idx].copy()
        if np.abs(mode_vec).max() > 0 and mode_vec[np.argmax(np.abs(mode_vec))] < 0:
            mode_vec *= -1.0

        disp = mode_vector_to_displacement(
            atoms=current,
            mode_vec=mode_vec,
            active_indices=active_indices,
            normalize_to=step_size,
        )

        nxt = current.copy()
        nxt.set_positions(current.get_positions() + disp)
        manifold_path.append(nxt)
        current = nxt

    return manifold_path, np.asarray(eig_history)


def align_atoms_to_reference(atoms: Atoms, ref_atoms: Atoms) -> Atoms:
    """Kabsch alignment (rotation only, no scaling)."""
    P = np.asarray(atoms.get_positions(), dtype=float)
    Q = np.asarray(ref_atoms.get_positions(), dtype=float)

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    C = Pc.T @ Qc
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt

    aligned = atoms.copy()
    aligned.set_positions(Pc @ U + Q.mean(axis=0))
    return aligned


# -----------------------------------------------------------------------------
# Manifold visualization
# -----------------------------------------------------------------------------

def plot_manifold_superposition(
    manifold_path: Sequence[Atoms],
    reference_atoms: Optional[Atoms] = None,
    title: str = "Manifold superposition",
    save_path: Optional[str] = None,
):
    if reference_atoms is None:
        reference_atoms = manifold_path[0]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ref = reference_atoms.copy()
    ref_pos = ref.get_positions()

    for atoms in manifold_path:
        a = align_atoms_to_reference(atoms, ref)
        pos = a.get_positions()
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.15, linewidth=0.8)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10, alpha=0.15)

    ax.scatter(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], s=40)
    for i in range(len(ref)):
        ax.text(ref_pos[i, 0], ref_pos[i, 1], ref_pos[i, 2], str(i), fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    return fig, ax


# -----------------------------------------------------------------------------
# Correlation plots
# -----------------------------------------------------------------------------

def _make_annotation_labels_global(pairs: Sequence[Tuple[int, int]]):
    return [f"({i}, {j})" for i, j in pairs]


def _make_annotation_labels_local(local_labels, pairs: Sequence[Tuple[int, int]]):
    out = []
    for i, j in pairs:
        out.append(f"({local_labels[i]}, {local_labels[j]})")
    return out


def _decorate_scatter_for_hover(fig, artist, labels, descriptor_mode: str):
    _init_hover_state(fig, [artist], [labels], descriptor_mode=descriptor_mode)
    return fig


def plot_pairwise_correlation(
    atoms_list: Sequence[Atoms],
    descriptor_fn_A,
    descriptor_fn_B,
    descriptor_mode: str = "global",
    n_atoms_per_config: Optional[int] = None,
    scale_A: Optional[float] = None,
    scale_B: Optional[float] = None,
    normalize: bool = True,
    perturb: float = 0.02,
    n_noise_samples: int = 200,
    max_pairs: int = 200000,
    bins: int = 200,
    n_procs: int = 1,
    seed: int = 0,
    title: str = "Correlation plot",
    save_path: Optional[str] = None,
    save_interactive_path: Optional[str] = None,
):
    
    """
    Plot a 2D correlation map between pairwise descriptor distances computed from
    two descriptors A and B.

    The function compares all sampled pairs of structures (global mode) or local
    atomic environments (local mode). For each pair (i, j), it computes:

        dA = ||F_A(i) - F_A(j)|| / scale_A
        dB = ||F_B(i) - F_B(j)|| / scale_B

    and shows the distribution of these pair distances as a 2D histogram
    (colormap). Optionally, it also attaches hover labels to sampled pairwise
    points.

    Parameters
    ----------
    atoms_list : Sequence[Atoms]
        List of ASE Atoms objects to compare.

    descriptor_fn_A : callable
        Function that takes one Atoms object and returns descriptor A.
        For local mode, it must return a list of local fingerprints when called
        with centers=... .

    descriptor_fn_B : callable
        Function that takes one Atoms object and returns descriptor B.
        Must have the same local/global behavior as descriptor_fn_A.

    descriptor_mode : str, default="global"
        Either "global" or "local".
        - "global": compare whole-structure descriptors across structures.
        - "local": compare local atomic environments across sampled centers.

    n_atoms_per_config : int or None, default=None
        Local mode only. Number of atoms to randomly sample per structure.
        If None, all atoms in each structure are used.

    scale_A : float or None, default=None
        Optional normalization scale for descriptor A distances.
        If None and normalize=True, the scale is estimated automatically from
        random perturbations of the structures.

    scale_B : float or None, default=None
        Same as scale_A, but for descriptor B.

    normalize : bool, default=True
        If True, divide descriptor distances by a descriptor-specific noise scale
        estimated from random small displacements.
        This makes different descriptors easier to compare on the same plot.

    perturb : float, default=0.02
        Size of the random Cartesian perturbation used to estimate the noise scale.
        Each atomic position is displaced randomly by up to approximately this
        amount in Angstrom.
        Larger values measure a more global structural change; smaller values
        measure a more local linear response.

    n_noise_samples : int, default=200
        Number of random perturbed structures used to estimate the noise scale.
        Larger values give a more stable estimate, but take longer.

    max_pairs : int, default=200000
        Maximum number of unique pairs to use when building the plot.
        If the full number of possible pairs is larger than this, the function
        randomly subsamples unique pairs without replacement.
        This controls the computational cost, memory use, and density of the plot.

    bins : int, default=200
        Number of histogram bins in each direction for the 2D histogram.
        Larger values give finer resolution but a sparser, noisier heatmap.

    n_procs : int, default=1
        Number of processes to use for descriptor evaluation.
        Use 1 for serial execution.
        Larger values parallelize descriptor computation across structures
        (global mode) or across configurations (local mode).

    seed : int, default=0
        Random seed used for pair subsampling, atom subsampling, and noise-scale
        estimation.

    title : str, default="Correlation plot"
        Plot title.

    save_path : str or None, default=None
        If given, save the visible plot as an image to this path.

    show_indices : bool, default=False
        If True, enable pair labels for hover or optional static annotations.
        In local mode, labels are pairs of local environment identifiers.
        In global mode, labels are pairs of structure indices.

    save_interactive_path : str or None, default=None
        If given, save an interactive pickled figure object to this path.
        Loading this pickle later and calling .show() will restore hover labels.

    Returns
    -------
    dA : np.ndarray
        Pairwise distances for descriptor A.

    dB : np.ndarray
        Pairwise distances for descriptor B.

    scale_A : float
        Noise scale used to normalize descriptor A distances.

    scale_B : float
        Noise scale used to normalize descriptor B distances.

    pairs : list[tuple[int, int]]
        List of sampled pair index pairs used to build the plot.
    """

    descriptor_mode = descriptor_mode.lower()
    if descriptor_mode not in {"global", "local"}:
        raise ValueError("descriptor_mode must be 'global' or 'local'")

    if descriptor_mode == "global":
        fps_A_raw = compute_global_descriptors(atoms_list, descriptor_fn_A, n_procs=n_procs)
        fps_B_raw = compute_global_descriptors(atoms_list, descriptor_fn_B, n_procs=n_procs)
        centers_list = None
    else:
        centers_list = _sample_centers_per_config(
            atoms_list, n_atoms_per_config=n_atoms_per_config, seed=seed
        )
        fps_A_raw, centers_list = compute_local_descriptors(
            atoms_list,
            descriptor_fn_A,
            centers_list=centers_list,
            n_procs=n_procs,
            seed=seed,
        )
        fps_B_raw, _ = compute_local_descriptors(
            atoms_list,
            descriptor_fn_B,
            centers_list=centers_list,
            n_procs=n_procs,
            seed=seed + 1,
        )
        if len(fps_A_raw) != len(fps_B_raw):
            raise RuntimeError("Local descriptor counts do not match. Use the same sampled centers.")
        env_labels = []
        for cfg_idx, centers in enumerate(centers_list):
            env_labels.extend([(cfg_idx, int(c)) for c in centers])

    max_len_A = max(np.asarray(fp).size for fp in fps_A_raw)
    max_len_B = max(np.asarray(fp).size for fp in fps_B_raw)

    fps_A = [_pad_or_truncate(fp, max_len_A) for fp in fps_A_raw]
    fps_B = [_pad_or_truncate(fp, max_len_B) for fp in fps_B_raw]

    if normalize:
        if scale_A is None:
            if descriptor_mode == "global":
                scale_A = _compute_noise_scale_global(
                    atoms_list, descriptor_fn_A, max_len_A, perturb=perturb, n_samples=n_noise_samples, seed=seed
                )
            else:
                scale_A = _compute_noise_scale_local(
                    atoms_list, descriptor_fn_A, centers_list, max_len_A, perturb=perturb, n_samples=n_noise_samples, seed=seed
                )
        if scale_B is None:
            if descriptor_mode == "global":
                scale_B = _compute_noise_scale_global(
                    atoms_list, descriptor_fn_B, max_len_B, perturb=perturb, n_samples=n_noise_samples, seed=seed + 1
                )
            else:
                scale_B = _compute_noise_scale_local(
                    atoms_list, descriptor_fn_B, centers_list, max_len_B, perturb=perturb, n_samples=n_noise_samples, seed=seed + 1
                )
    else:
        scale_A = 1.0
        scale_B = 1.0

    n = len(fps_A)
    if n < 2:
        raise ValueError("Need at least two structures/environments.")

    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    if max_pairs is None or max_pairs >= total_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        seen = set()
        while len(pairs) < max_pairs:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            key = (int(i), int(j))
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    dA = np.empty(len(pairs), dtype=float)
    dB = np.empty(len(pairs), dtype=float)
    for k, (i, j) in enumerate(pairs):
        dA[k] = np.linalg.norm(fps_A[i] - fps_A[j]) / scale_A
        dB[k] = np.linalg.norm(fps_B[i] - fps_B[j]) / scale_B
    if descriptor_mode == "global":
        pair_labels = [(int(i), int(j)) for i, j in pairs]
    else:
        pair_labels = [(env_labels[i], env_labels[j]) for i, j in pairs]

    fig, ax = plt.subplots(figsize=(4, 3))

    # visible plot stays unchanged
    h = ax.hist2d(dA, dB, bins=bins, norm=LogNorm(), cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="pair count")

    hover_artists = []
    if save_interactive_path is not None:
        hover_sc = _make_hover_scatter(ax, dA, dB, pair_labels)
        hover_artists.append(hover_sc)

    ax.set_xlabel(r"$d_A / \sigma_A$")
    ax.set_ylabel(r"$d_B / \sigma_B$")
    ax.set_title(title)
    plt.tight_layout()

    out = InteractiveFigure(fig, hover_artists=hover_artists)

    if save_interactive_path is not None:
        out.save_pickle(save_interactive_path)

    if save_path is not None:
        fig.savefig(save_path, dpi=600)

    return dA, dB, scale_A, scale_B, pairs


def plot_local_pairwise_correlation(*args, **kwargs):
    kwargs["descriptor_mode"] = "local"
    return plot_pairwise_correlation(*args, **kwargs)


def plot_global_pairwise_correlation(*args, **kwargs):
    kwargs["descriptor_mode"] = "global"
    return plot_pairwise_correlation(*args, **kwargs)


# -----------------------------------------------------------------------------
# Sensitivity spectrum plotting
# -----------------------------------------------------------------------------

def _make_spectrum_labels(
    step_indices: Sequence[int],
    mode_index: int,
    evals: Sequence[float],
    descriptor_name: str = "",
):
    labels = []
    for step_idx, lam in zip(step_indices, evals):
        if descriptor_name:
            labels.append(f"{descriptor_name} | s{step_idx}, m{mode_index}, λ={lam:.2e}")
        else:
            labels.append(f"s{step_idx}, m{mode_index}, λ={lam:.2e}")
    return labels


def plot_sensitivity_spectra_along_manifold(
    manifold_path: Sequence[Atoms],
    descriptor_fns: Dict[str, Callable],
    active_indices: Optional[Sequence[int]] = None,
    eps: float = 1e-4,
    descending: bool = False,
    n_procs: int = 1,
    n_show: int = 10,
    skip_rigid: int = 6,
    title: str = "Sensitivity eigenvalues along manifold",
    save_path: Optional[str] = None,
    save_interactive_path: Optional[str] = None,
):
    names = list(descriptor_fns.keys())
    n = len(names)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharex=True
    )
    axes = np.asarray(axes).reshape(-1)

    hover_artists = []

    for ax, (name, fn) in zip(axes, descriptor_fns.items()):
        tracks = []
        step_indices = []

        for step_idx, atoms in enumerate(manifold_path):
            evals, _, _, _ = sensitivity_eigendecomposition(
                atoms=atoms,
                descriptor_fn=fn,
                active_indices=active_indices,
                descending=descending,
                eps=eps,
                n_procs=n_procs,
                normalize=True,
            )
            if descending:
                # rigid near-zeros are at the tail, take first n_show large ones
                tracks.append(evals[:n_show])  
            else:
                tracks.append(evals[skip_rigid : skip_rigid + n_show])
            tracks.append(evals[skip_rigid : skip_rigid + n_show])
            step_indices.append(step_idx)

        tracks = np.asarray(tracks)
        step_indices = np.asarray(step_indices, dtype=int)

        for i in range(tracks.shape[1]):
            y = tracks[:, i]
            ax.plot(step_indices, y, "-o", lw=1.2, ms=3)

            labels = [
                f"{name} | step={s}, mode={skip_rigid + i + 1}, λ={val:.2e}"
                for s, val in zip(step_indices, y)
            ]

            if save_interactive_path is not None:
                # Invisible hover layer; does not alter visible style.
                sc = ax.scatter(
                    step_indices,
                    y,
                    s=20,
                    alpha=0.0,
                    linewidths=0.0,
                    edgecolors="none",
                    picker=True,
                )
                sc._descriptor_hover_labels = labels
                hover_artists.append(sc)

        ax.set_title(name)
        ax.set_xlabel("Trajectory step")
        ax.set_ylabel("Normalized eigenvalue")
        ax.set_ylim(1e-16, 1.1)

    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    interactive = InteractiveFigure(fig, hover_artists=hover_artists)

    if save_interactive_path is not None:
        interactive.save_pickle(save_interactive_path)

    if save_path is not None:
        fig.savefig(save_path, dpi=600)

    return fig, axes


def plot_sensitivity_spectra(
    atoms: Atoms,
    descriptor_fns: Dict[str, Callable],
    active_indices: Optional[Sequence[int]] = None,
    descending: bool = False,
    eps: float = 1e-4,
    n_procs: int = 1,
    normalize: bool = True,
    title: str = "Sensitivity eigenvalue spectra",
    save_path: Optional[str] = None,
    save_interactive_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    hover_artists = []

    for name, fn in descriptor_fns.items():
        evals, _, _, _ = sensitivity_eigendecomposition(
            atoms=atoms,
            descriptor_fn=fn,
            active_indices=active_indices,
            descending=descending,
            eps=eps,
            n_procs=n_procs,
            normalize=normalize,
        )

        x = np.arange(len(evals), dtype=int)
        y = evals + 1e-18

        ax.semilogy(x, y, label=name, marker="o", ms=3, lw=1.2)

        labels = [f"{name} | mode={k + 1}, λ={val:.2e}" for k, val in enumerate(evals)]

        if save_interactive_path is not None:
            sc = ax.scatter(
                x,
                y,
                s=20,
                alpha=0.0,
                linewidths=0.0,
                edgecolors="none",
                picker=True,
            )
            sc._descriptor_hover_labels = labels
            hover_artists.append(sc)

    ax.set_xlabel("Mode index")
    ax.set_ylabel("Normalized eigenvalue")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.tight_layout()

    interactive = InteractiveFigure(fig, hover_artists=hover_artists)

    if save_interactive_path is not None:
        interactive.save_pickle(save_interactive_path)

    if save_path is not None:
        fig.savefig(save_path, dpi=600)

    return fig, ax


# -----------------------------------------------------------------------------
# Miscellaneous helpers
# -----------------------------------------------------------------------------

def random_perturbations(
    atoms: Atoms,
    n_samples: int = 100,
    rms_disp: float = 0.03,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    out = []
    R0 = atoms.get_positions()

    for _ in range(int(n_samples)):
        delta = rng.normal(size=R0.shape)
        delta -= delta.mean(axis=0)
        rms = np.sqrt(np.mean(np.sum(delta**2, axis=1)))
        if rms > 0:
            delta *= (float(rms_disp) / rms)
        a = atoms.copy()
        a.set_positions(R0 + delta)
        out.append(a)
    return out


def four_body_energy(atoms: Atoms) -> float:
    """Simple torsion-like target: sum over cos^2(dihedral) for all 4-tuples."""
    R = np.asarray(atoms.get_positions(), dtype=float)
    n = len(R)
    if n < 4:
        return 0.0

    def dihedral(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= (np.linalg.norm(b1) + 1e-18)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.arctan2(y, x)

    E = 0.0
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    phi = dihedral(R[i], R[j], R[k], R[l])
                    E += np.cos(phi) ** 2
    return float(E)


__all__ = [
    "OMDescriptor",
    "SOAPDescriptor",
    "ACSFDescriptor",
    "MACEDescriptor",
    "SingleCenterDescriptor",
    "descriptor_catalog",
    "compute_local_descriptors",
    "compute_global_descriptors",
    "plot_pairwise_correlation",
    "plot_local_pairwise_correlation",
    "plot_global_pairwise_correlation",
    "finite_difference_jacobian",
    "sensitivity_eigendecomposition",
    "mode_vector_to_displacement",
    "trace_quasi_constant_manifold",
    "plot_manifold_superposition",
    "plot_sensitivity_spectra_along_manifold",
    "plot_sensitivity_spectra",
    "random_perturbations",
    "four_body_energy",
]
