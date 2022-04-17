import json,os

# create with respect to gpu num
with open("CVPR_2022_NAS_Track1_test.json") as json_file:
    arch_dict = json.load(json_file)
    arch_code_idx_dict = {}
    for arch_idx in list(arch_dict.keys()):
        arch_code_idx_dict[arch_dict[arch_idx]["arch"]] = [arch_idx, 0.0]

    for i in range(0, 8):
        with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_gpu_{}.json".format(i)),
                  "w") as submit_json_file:
            json.dump(arch_code_idx_dict, submit_json_file)

# write demo
with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_gpu_{}.json".format(0)),
                  "r") as submit_json_file:
            # json.dump(arch_code_idx_dict, submit_json_file)
            arch_dict = json.load(submit_json_file)
            if "1252233313000000223212122253130000000000006515000000" in list(arch_dict.keys()):
                arch_dict["1252233313000000223212122253130000000000006515000000"][1] = 120.0
            arch_dict["1245211121000000133323130054242414240000007424000000"][1] = 69.0
            # json.dump(arch_dict, submit_json_file)
with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_gpu_{}.json".format(0)),
                  "w") as submit_json_file:
    json.dump(arch_dict, submit_json_file)
