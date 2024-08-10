# -*- coding: utf-8 -*-

from typing import Optional


def update_depth_and_ssp(
    h_w: float, z_values: list, c_p_values: list
) -> tuple[list, list]:
    z_max = z_values[-1]
    if h_w > z_max:
        z_values.append(h_w)
        c_p_values.append(c_p_values[-1])
    if h_w < z_max:
        rm_ind = list(filter(lambda x: z_values[x] >= h_w, range(len(z_values))))
        z_values = [el for i, el in enumerate(z_values) if i not in rm_ind] + [h_w]
        c_p_values = [el for i, el in enumerate(c_p_values) if i not in rm_ind] + [
            c_p_values[rm_ind[-1]]
        ]
    # Handle cases where the two final depths are nearly equal and will
    # appear so in the ENV file due to lower precision.
    if z_values[-1] - z_values[-2] < 0.01:
        return z_values[:-1], c_p_values[:-1]
    return z_values, c_p_values


def update_receiver_depth(dz: float, rec_z: list) -> list:
    return [el + dz for el in rec_z]


def update_pivot_depth(dz: float, z_pivot: Optional[float] = None) -> float:
    if z_pivot is None:
        return None
    return z_pivot + dz


def update_sediment(dz: float, z_values: list) -> list:
    return [el + dz for el in z_values]


def format_h_w(
    fixed_parameters: dict, search_parameters: dict, h_w: Optional[float] = None
) -> tuple[dict, dict, float]:
    water_data = fixed_parameters["layerdata"][0]
    if h_w is None:
        h_w = water_data["z"][-1]
    else:
        z_values = water_data["z"]
        c_p_values = water_data["c_p"]
        dz = h_w - z_values[-1]

        z_values, c_p_values = update_depth_and_ssp(h_w, z_values, c_p_values)

        # Adjust receiver depths
        fixed_parameters["rec_z"] = update_receiver_depth(dz, fixed_parameters["rec_z"])
        fixed_parameters["z_pivot"] = update_pivot_depth(
            dz, fixed_parameters["z_pivot"]
        )

        water_data["z"] = z_values
        water_data["c_p"] = c_p_values
        fixed_parameters["layerdata"][0] = water_data

        # Adjust sediment depths and SSP
        sediment_data = fixed_parameters["layerdata"][1]
        z_s_values = update_sediment(dz, sediment_data["z"])
        sediment_data["z"] = z_s_values
        fixed_parameters["layerdata"][1] = sediment_data

        # Adjust mudrock depths and SSP
        bottom_data = fixed_parameters["layerdata"][2]
        z_b_values = update_sediment(dz, bottom_data["z"])
        bottom_data["z"] = z_b_values
        fixed_parameters["layerdata"][2] = bottom_data

    return fixed_parameters, search_parameters, h_w


def format_h_sed(
    fixed_parameters: dict,
    search_parameters: dict,
    h_w: float,
    h_sed: Optional[float] = None,
) -> tuple[dict, dict]:
    sediment_data = fixed_parameters["layerdata"][1]
    if h_sed is not None:
        sediment_data["z"] = [h_w, h_w + h_sed]
        fixed_parameters["layerdata"][1] = sediment_data
    return fixed_parameters, search_parameters


def format_c_p_sed_top(
    fixed_parameters: dict,
    search_parameters: dict,
    c_p_sed_top: Optional[float] = None,
    c_p_sed_bot: Optional[float] = None,
    dc_p_sed: float = 0.0,
) -> tuple[dict, dict]:
    sediment_data = fixed_parameters["layerdata"][1]
    if c_p_sed_top is not None:
        # NOTE: If c_p_sed_top is specified without an accompanying SSP
        # gradient, the sediment sound speed is fixed to a constant value.
        sediment_data["c_p"] = [c_p_sed_top, c_p_sed_top + dc_p_sed]
        fixed_parameters["layerdata"][1] = sediment_data
    if c_p_sed_bot is not None:
        sediment_data["c_p"] = [c_p_sed_top, c_p_sed_bot]
        fixed_parameters["layerdata"][1] = sediment_data
    return fixed_parameters, search_parameters


def format_a_p_sed_top(
    fixed_parameters: dict,
    search_parameters: dict,
    a_p_sed: Optional[float] = None,
) -> tuple[dict, dict]:
    if a_p_sed is not None:
        sediment_data = fixed_parameters["layerdata"][1]
        sediment_data["a_p"] = a_p_sed
        fixed_parameters["layerdata"][1] = sediment_data
    return fixed_parameters, search_parameters


def format_rho_sed_top(
    fixed_parameters: dict,
    search_parameters: dict,
    rho_sed: Optional[float] = None,
) -> tuple[dict, dict]:
    if rho_sed is not None:
        sediment_data = fixed_parameters["layerdata"][1]
        sediment_data["rho"] = rho_sed
        fixed_parameters["layerdata"][1] = sediment_data
    return fixed_parameters, search_parameters


def format_parameters(
    freq: float, title: str, fixed_parameters: dict, search_parameters: dict
) -> dict:
    # Adjust water depth and sound speed profile.
    h_w = search_parameters.pop("h_w", None)
    fixed_parameters, search_parameters, h_w = format_h_w(
        fixed_parameters, search_parameters, h_w
    )
    # Adjust sediment depth
    h_sed = search_parameters.pop("h_sed", None)
    fixed_parameters, search_parameters = format_h_sed(
        fixed_parameters, search_parameters, h_w, h_sed
    )

    # Adjust sediment sound speed
    c_p_sed_top = search_parameters.pop("c_p_sed_top", None)
    c_p_sed_bot = search_parameters.pop("c_p_sed_bot", None)
    dc_p_sed = search_parameters.pop("dc_p_sed", 0.0)
    fixed_parameters, search_parameters = format_c_p_sed_top(
        fixed_parameters, search_parameters, c_p_sed_top, c_p_sed_bot, dc_p_sed
    )

    # Adjust sediment attenuation
    a_p_sed = search_parameters.pop("a_p_sed", None)
    fixed_parameters, search_parameters = format_a_p_sed_top(
        fixed_parameters, search_parameters, a_p_sed
    )

    # Adjust sediment density
    rho_sed = search_parameters.pop("rho_sed", None)
    fixed_parameters, search_parameters = format_rho_sed_top(
        fixed_parameters, search_parameters, rho_sed
    )

    return fixed_parameters | {"freq": freq, "title": title} | search_parameters
