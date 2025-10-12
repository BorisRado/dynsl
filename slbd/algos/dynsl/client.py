import os
import psutil
from typing import Dict
try:
    from kubernetes import client, config
except:
    pass

import torch
from flwr.common import Config, Scalar

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS
)



def get_node_name():
    # Load in-cluster config (works from inside any Pod)
    config.load_incluster_config()

    # Create CoreV1 API client
    v1 = client.CoreV1Api()

    # Get current pod name and namespace
    pod_name = os.getenv("HOSTNAME")
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    # Query the Kubernetes API for this pod
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

    # Return the node name
    return pod.spec.node_name


class DynSLClientAdapter(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        # no limits if client has GPU, otherwise ... how to determine max layers the client can afford? TODO
        print("Client GPU?", torch.cuda.is_available())
        device_type = os.environ.get("COLEXT_DEVICE_TYPE", "")
        if device_type == "JetsonNano":
            max_layers = self.whole_model.meaningful_split_points[2]
        elif device_type == "JetsonXavierNX":
            max_layers = self.whole_model.meaningful_split_points[3]
        elif device_type == "JetsonOrinNano":
            max_layers = self.whole_model.meaningful_split_points[3]
        elif torch.cuda.is_available():
            max_layers = 1e4
        else:
            ram = psutil.virtual_memory().total / (1 << 30)
            print(f"Client NO GPU has {ram}GB of RAM -- {device_type}")
            # if ram < 6:
            max_layers = self.whole_model.meaningful_split_points[1]
        try:
            node_name = get_node_name()
        except:
            node_name = ""
        return {
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: device_type,
            MAX_NUMBER_OF_CLIENT_LAYERS: max_layers,
            "node_name": node_name
        }
