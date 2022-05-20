def convert_subnet_settings_to_arch_code(subnet_settings):
    """
    :param subnet_settings: dict of settings
    {"e": xxxx, "d": xxxxx}
    # e = [0.9, 0.9, 0.94, 0.8, 0.85...]
    # d = [5, 5, 8, 5]
    :return:
    """
    e_map = {'1':1.0, '2':0.95, '3':0.9, '4':0.85, '5':0.8, '6':0.75, '7':0.7}
    e_inverse_map = {}
    for key in list(e_map.keys()):
        e_inverse_map[e_map[key]] = key
    depth_max_list = [5, 5, 8, 5]

    code_string = "1"

    # depth encode
    for d_item in subnet_settings["d"]:
        code_string += str(d_item)

    # expand encode
    # expand_1d_list = [subnet_settings["e"][0]]
    # for item_list in subnet_settings["e"][1:]:
    #     expand_1d_list += item_list

    # code_string += str(subnet_settings["e"][0]) # stem block
    # depth_cnt = 1
    # for stage_idx, d_item in enumerate(subnet_settings["d"]):
    #     for depth_num in range(depth_max_list[stage_idx]): # fill in 0s, if depth setting is less than max depth
    #         if depth_num < d_item:
    #             code_string += str(subnet_settings["e"][depth_cnt+depth_num][0]) + \
    #                             str(str(subnet_settings["e"][depth_cnt+depth_num][1]))
    #         else:
    #             code_string += "00"
    #     depth_cnt += depth_max_list[stage_idx]

    code_string += str(e_inverse_map[subnet_settings["e"][0]])  # stem block
    depth_cnt = 1
    for stage_idx, d_item in enumerate(subnet_settings["d"]):
        for depth_num in range(depth_max_list[stage_idx]):  # fill in 0s, if depth setting is less than max depth
            if depth_num < d_item:
                code_string += str(e_inverse_map[subnet_settings["e"][depth_cnt + depth_num*2]]) + \
                               str(e_inverse_map[subnet_settings["e"][depth_cnt + depth_num*2+1]])
            else:
                code_string += "00"
        depth_cnt += depth_max_list[stage_idx] * 2

    return code_string


if __name__ == "__main__":
    subnet_settings = {
        "e":[0.85, 0.85, 1.0, 1.0, 0.85, 0.8, 0.95, 0.85, 0.95, 0.8, 0.8, 1.0, 0.95, 0.9, 0.9, 0.95, 0.75, 0.8, 0.95, 0.8, 0.8, 0.9, 0.85, 1.0, 0.95, 0.8, 0.7, 0.7, 1.0, 0.75, 0.75, 0.85, 0.75, 1.0, 0.75, 0.75, 0.85, 0.8, 0.9, 0.85, 0.95, 0.9, 0.7, 0.9, 0.85, 0.95, 0.95],
        "d":[3, 3, 7, 2]
    }
    code_string = convert_subnet_settings_to_arch_code(subnet_settings)
    print(code_string)
    print(len(code_string))