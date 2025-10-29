import os
import time

try:
    from kubernetes import client as k8s_client, config as k8s_config
except:
    print("KUBERNETES NOT INSTALLED -- LIKELY NOT RUNNING ON COLEXT???")
    k8s_config = None


def get_node_name():
    if k8s_config is None:
        return ""
    # Load in-cluster config (works from inside any Pod)
    k8s_config.load_incluster_config()

    # Create CoreV1 API client
    v1 = k8s_client.CoreV1Api()

    # Get current pod name and namespace
    pod_name = os.getenv("HOSTNAME")
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    # Query the Kubernetes API for this pod
    pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

    # Return the node name
    return pod.spec.node_name


def get_all_pods_and_nodes():
    if k8s_config is None:
        return {}

    k8s_config.load_incluster_config()
    v1 = k8s_client.CoreV1Api()

    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        namespace = f.read().strip()

    while True:
        pods = v1.list_namespaced_pod(namespace=namespace)
        pods_dict = {
            p.metadata.name: p.spec.node_name
            for p in pods.items
            if p.status.phase == "Running"
        }
        if len(pods_dict) < int(os.environ["COLEXT_N_CLIENTS"])+1: # +1 for the server
            print("Not enough pods detected. Continuing...")
            time.sleep(5.0)
        else:
            return pods_dict
