# -*- coding: utf-8 -*-


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
    return z_values, c_p_values


def update_receiver_depth(dz: float, rec_z: list) -> list:
    return [el + dz for el in rec_z]


def update_pivot_depth(dz: float, z_pivot: float) -> float:
    if z_pivot is None:
        return None
    return z_pivot + dz


def update_sediment(dz: float, z_values: list) -> list:
    return [el + dz for el in z_values]


def format_parameters(
    freq: float, title: str, fixed_parameters: dict, search_parameters: dict
) -> dict:

    c1 = search_parameters.pop("c1", None)
    dc1 = search_parameters.pop("dc1", None)
    dc2 = search_parameters.pop("dc2", None)
    dc3 = search_parameters.pop("dc3", None)
    dc4 = search_parameters.pop("dc4", None)
    dc5 = search_parameters.pop("dc5", None)
    dc6 = search_parameters.pop("dc6", None)
    dc7 = search_parameters.pop("dc7", None)
    dc8 = search_parameters.pop("dc8", None)
    dc9 = search_parameters.pop("dc9", None)
    # dc10 = search_parameters.pop("dc10", None)

    if c1 is not None:
        water_data = fixed_parameters["layerdata"][0]["c_p"]
        c2 = c1 + dc1
        c3 = c2 + dc2
        c4 = c3 + dc3
        c5 = c4 + dc4
        c6 = c5 + dc5
        c7 = c6 + dc6
        c8 = c7 + dc7
        c9 = c8 + dc8
        c10 = c9 + dc9
        # c11 = c10 + dc10
        c_p_values = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
        fixed_parameters["layerdata"][0]["c_p"] = c_p_values

    # Adjust water depth and sound speed profile.
    h_w = search_parameters.pop("h_w", None)
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

        # # Adjust sediment depths and SSP
        # sediment_data = fixed_parameters["layerdata"][1]
        # z_s_values = update_sediment(dz, sediment_data["z"])
        # sediment_data["z"] = z_s_values
        # fixed_parameters["layerdata"][1] = sediment_data

    # # Adjust bottom sound speed
    # bot_c_p = search_parameters.pop("bot_c_p", None)
    # if bot_c_p is not None:
    #     fixed_parameters["bot_c_p"] = bot_c_p


    # # Adjust sediment depth
    # h_s = search_parameters.pop("h_s", None)
    # sediment_data = fixed_parameters["layerdata"][1]
    # if h_s is None:
    #     h_s = sediment_data["z"][-1] - sediment_data["z"][0]
    # else:
    #     sediment_data["z"] = [h_w, h_w + h_s]
    #     fixed_parameters["layerdata"][1] = sediment_data

    # # Adjust sediment sound speed
    # c_s = search_parameters.pop("c_s", None)
    # if c_s is None:
    #     c_s = sediment_data["c_p"][0]
    # else:
    #     # NOTE: If c_s is specified without an accompanying SSP gradient,
    #     # the sediment sound speed is fixed to a constant value.
    #     sediment_data["c_p"] = [c_s, c_s]
    #     fixed_parameters["layerdata"][1] = sediment_data

    # # Adjust sediment sound speed gradient
    # dcdz_s = search_parameters.pop("dcdz_s", None)
    # if dcdz_s is None:
    #     dcdz_s = (sediment_data["c_p"][1] - sediment_data["c_p"][0]) / h_s
    # sediment_data["c_p"] = [c_s, c_s + dcdz_s * h_s]
    # fixed_parameters["layerdata"][1] = sediment_data

    return fixed_parameters | {"freq": freq, "title": title} | search_parameters
