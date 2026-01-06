import argparse

from jinja2 import Template


"""
Simple script to create a colext config for a given experiment. It injects the fl_algorithm,
the data configuration, and any further configuration
"""
COLEXT_TEMPLATE = "exp/colext/colext_template.yaml"
COLEXT_OUT_CONFIG = "colext_config.yaml"

def generate():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, type=str, help="FL algorithm to be tested",
                        choices=["splitfed_v1", "splitfed_v2", "ares", "dynsl_v2", "oracle"])
    parser.add_argument("--model-name", required=True, type=str, help="Model to be trained")
    parser.add_argument("--num-rounds", required=False, type=int, default=1, help="Number of training rounds")
    parser.add_argument("--num-partitions", required=False, type=int, default=10, help="Number of partitions of the training dataset")
    parser.add_argument("--experiment-name", required=False, type=str, default="colext", help="Name of experiment on wandb")

    # common arguments for all algorithms
    parser.add_argument("--num-server-models", required=True, type=int, help="Number of models to be trained on the server")

    # number of devices
    parser.add_argument("--num-jao", required=False, type=int, default=2, help="Number of agx devices to use")
    parser.add_argument("--num-jn", required=False, type=int, default=6, help="Number of jn devices to use")
    parser.add_argument("--num-jon", required=False, type=int, default=4, help="Number of jon devices to use")
    parser.add_argument("--num-jxn", required=False, type=int, default=2, help="Number of jxn devices to use")
    parser.add_argument("--num-lp", required=False, type=int, default=4, help="Number of lp devices to use")
    parser.add_argument("--num-op", required=False, type=int, default=7, help="Number of op devices to use")

    # algorithm specific configuration -- SPLITFED
    parser.add_argument("--splitfed-num-client-layers", required=False, type=int, default=4, help="Number of layer trained on the client [SPLITFED]")

    # algorithm specific configuration -- DYNSL
    parser.add_argument("--dynsl-reallocation-frequency", required=False, type=float, default=8.0, help="... [DYNSL]")
    parser.add_argument("--dynsl-max-client-layers", required=False, type=int, default=-1, help="... [DYNSL]")
    parser.add_argument("--dynsl-gamma", required=False, type=float, default=8.0, help="... [DYNSL]")
    parser.add_argument("--dynsl-cache-client-configuration", required=False, type=str, default="false", choices=["true", "false"], help="... [DYNSL]")
    parser.add_argument("--dynsl-layer-allocation-optimizer", required=False, type=str, default="genetic", choices=["genetic"], help="... [DYNSL]")
    parser.add_argument("--dynsl-forking", required=False, type=str, default="false", choices=["true", "false"], help="... [DYNSL]")

    # algorithm specific configuration -- ORACLE
    parser.add_argument("--oracle-nl-jao", required=False, type=int, default=-1, help="... [ORACLE]")
    parser.add_argument("--oracle-nl-jn", required=False, type=int, default=-1, help="... [ORACLE]")
    parser.add_argument("--oracle-nl-jon", required=False, type=int, default=-1, help="... [ORACLE]")
    parser.add_argument("--oracle-nl-jxn", required=False, type=int, default=-1, help="... [ORACLE]")
    parser.add_argument("--oracle-nl-lp", required=False, type=int, default=-1, help="... [ORACLE]")
    parser.add_argument("--oracle-nl-op", required=False, type=int, default=-1, help="... [ORACLE]")

    # CONFIGURATION OF BACKGROUND ACTIVITY
    parser.add_argument("--background-processes", required=False, action="store_true")
    parser.add_argument("--bg-proc-delay", required=False, type=float, default=0., help="...")
    parser.add_argument("--bg-proc-duration", required=False, type=float, default=0., help="...")
    parser.add_argument("--bg-num-procs", required=False, type=int, default=0, help="...")
    parser.add_argument("--bg-dev-num-jao", required=False, type=int, default=0)
    parser.add_argument("--bg-dev-num-jn", required=False, type=int, default=0)
    parser.add_argument("--bg-dev-num-jon", required=False, type=int, default=0)
    parser.add_argument("--bg-dev-num-jxn", required=False, type=int, default=0)
    parser.add_argument("--bg-dev-num-lp", required=False, type=int, default=0)
    parser.add_argument("--bg-dev-num-op", required=False, type=int, default=0)

    args = parser.parse_args()
    args_dict = vars(args)

    if args.algorithm.startswith("splitfed"):
        args_dict["algo_config"] = f"algorithm.layer_allocator_init_args.num_client_layers={args.splitfed_num_client_layers}"

    if args.algorithm.startswith("dynsl"):
        predix_len = len("dynsl_")
        algo_args = []
        for key in args_dict:
            if not key.startswith("dynsl"):
                continue
            prefix = "algorithm."
            if key[predix_len:] != "forking":
                prefix += "layer_allocator_init_args."
            algo_args.append(f"{prefix}{key[predix_len:]}={args_dict[key]}")
        args_dict["algo_config"] = "\n      ".join(algo_args)

    if args.algorithm.startswith("oracle"):
        tmp = []
        for key, val in args_dict.items():
            if not key.startswith("oracle"):
                continue
            device_type_acronym = key.split("_")[-1]
            if args_dict[f"num_{device_type_acronym}"] == 0:
                continue
            assert val > 0
            device_type = {
                "jao": "JetsonAGXOrin",
                "jn": "JetsonNano",
                "jon": "JetsonOrinNano",
                "jxn": "JetsonXavierNX",
                "lp": "LattePandaDelta3",
                "op": "OrangePi5B",
            }[device_type_acronym]
            tmp.append(f"+algorithm.layer_allocator_init_args.device_mapping.{device_type}={val}")
        args_dict["algo_config"] = "\n      ".join(tmp)

    # generate a string with the number of devices to be appended to the config
    device_listing = " ".join([
        f"+{key}={args_dict[key]}"
        for key in ["num_jao", "num_jon", "num_jn", "num_lp", "num_jxn", "num_op"]
    ])
    args_dict["device_listing"] = device_listing

    if args.background_processes:
        # configure the background activity
        assert (
            args.bg_num_procs > 0 and
            args.bg_proc_duration > 0
        )
        bg_configuration = \
            "+client.init_kwargs='{bg_proc_duration:" + str(args.bg_proc_duration) + \
            ",bg_num_procs:" + str(args.bg_num_procs) + \
            ",bg_proc_delay:" + str(args.bg_proc_delay) + \
            "}'"
        # handle clients
        bg_devices_listing = "+bg_proc_devices='{"
        tot_devices_with_bg = 0
        for config in args_dict:
            if not config.startswith("bg_dev_num_"):
                continue
            if args_dict[config] == 0:
                continue
            bg_devices_listing += config.split("_")[-1] + ": " + str(args_dict[config]) + ", "
            tot_devices_with_bg += args_dict[config]
        bg_devices_listing = bg_devices_listing[:-2] + "}'"
        args_dict["bg_devices"] = bg_devices_listing
        args_dict["bg_configuration"] = bg_configuration

        assert tot_devices_with_bg > 0

    with open(COLEXT_TEMPLATE, "r") as file:
        template = Template(file.read())

    rendered_content = template.render(args_dict)

    with open(COLEXT_OUT_CONFIG, "w") as file:
        file.write(rendered_content)

    print("YAML file generated successfully.")


if __name__ == "__main__":
    generate()
