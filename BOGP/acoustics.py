#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from tqdm import tqdm

from tritonoa.io import read_ssp
from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer, snrdb_to_sigma, added_wng
from .utils import clean_up_kraken_files


class MatchedFieldProcessor:
    def __init__(self, K, parameters, atype="cbf"):
        self.K = K
        self.parameters = parameters
        self.atype = atype

    def __call__(self, parameters, scale=1):
        return self.evaluate_true(parameters)

    def __str__(self):
        return self.__class__.__name__

    def evaluate_true(self, parameters):
        p_rep = run_kraken(self.parameters | parameters)
        return abs(beamformer(self.K, p_rep, atype=self.atype).item())
        # return 10 * np.log10(beamformer(self.K, p_rep, atype=self.atype).item())

    def _get_name(self):
        return self.__class__.__name__


def run_mfp(parameters, mode, dr=None, dz=None, nr=None, nz=None):
    # dr [km]
    # dz [m]
    fixed_parameters = parameters["fixed_parameters"]
    search_parameters = parameters["search_parameters"]

    sigma = snrdb_to_sigma(fixed_parameters["snr"])
    p_rec = run_kraken(fixed_parameters)
    p_rec /= np.linalg.norm(p_rec)
    noise = added_wng(p_rec.shape, sigma=sigma, cmplx=True)
    p_rec += noise
    K = p_rec.dot(p_rec.conj().T)

    if nr is None:
        if dr is None:
            dr = 5 / 1e3
        rvec = np.arange(
            search_parameters[0]["bounds"][0],
            search_parameters[0]["bounds"][1] + dr,
            dr,
        )
    else:
        rvec = np.linspace(
            search_parameters[0]["bounds"][0], search_parameters[0]["bounds"][1], nr
        )

    if mode == "r":
        fixed_parameters.pop("rec_r")
        p_rep = run_kraken(fixed_parameters | {"rec_r": rvec})
        B = np.zeros((1, len(rvec)))
        for rr, r in enumerate(rvec):
            B[0, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()

        B /= np.max(np.abs(B))
        clean_up_kraken_files(".")
        return B, rvec

    elif mode == "l":
        [fixed_parameters.pop(item) for item in ["rec_r", "src_z"]]

        if nz is None:
            if dz is None:
                dz = 2
            zvec = np.arange(
                search_parameters[1]["bounds"][0], search_parameters[1]["bounds"][1], dz
            )
        else:
            zvec = np.linspace(
                search_parameters[1]["bounds"][0], search_parameters[1]["bounds"][1], nz
            )

        # p_rep = np.zeros((len(zvec), len(rvec), len(fixed_parameters["rec_z"])))
        B = np.zeros((len(zvec), len(rvec)))

        pbar = tqdm(
            zvec,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            desc="  MFP",
            leave=True,
            position=0,
            unit=" step",
        )

        for zz, z in enumerate(pbar):
            p_rep = run_kraken(fixed_parameters | {"src_z": z, "rec_r": rvec})
            for rr, r in enumerate(rvec):
                B[zz, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()

        B /= np.max(np.abs(B))
        clean_up_kraken_files(".")
        return B, rvec, zvec


# Load CTD data
# z_data, c_data, _ = read_ssp(
#     Path.cwd() / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
#     # Path.cwd() / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
# )
# z_data = np.append(z_data, 217).tolist()
# c_data = np.append(c_data, c_data[-1]).tolist()

# ENV_SWELLEX96 = {
#     # "title": "SWELLEX96_SIM",
#     # "tmpdir": "tmp",
#     # "model": "KRAKENC",
#     # Top medium
#     # Layered media
#     "layerdata": [
#         {"z": z_data, "c_p": c_data, "rho": 1},
#         {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
#         {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
#     ],
#     # Bottom medium
#     "bot_opt": "A",
#     "bot_c_p": 5200,
#     "bot_rho": 2.7,
#     "bot_a_p": 0.03,
#     # Speed constraints
#     "clow": 0,
#     "chigh": 1600,
#     # Receiver parameters
#     "rec_z": np.linspace(94.125, 212.25, 64),
#     # Source parameters
#     # "rec_r": RANGE_TRUE,
#     # "src_z": DEPTH_TRUE,
#     # "freq": FREQ,
# }
