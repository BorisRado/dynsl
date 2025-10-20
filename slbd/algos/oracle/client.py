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
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
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


class OracleNumpyClientAdapter(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        # no limits if client has GPU, otherwise ... how to determine max layers the client can afford? TODO
        print("Client GPU?", torch.cuda.is_available())
        device_type = os.environ.get("COLEXT_DEVICE_TYPE", "")
        try:
            node_name = get_node_name()
        except:
            node_name = ""
        return {
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: device_type,
            "node_name": node_name
        }
