def ares_compute_optimal_split_layer(
    client_fb_times, # fb indicates that it is the sum of the forward and backward time - it is cumulative from start to the considered layer
    server_fb_times, # fb indicates that it is the sum of the forward and backward time - it is layer-by-layer
    tensor_sizes,
    network_throughput_Mbps,
):
    """
    Compute optimal split vector for ARES based on training time minimization.

    Args:
        client_forward_times: List of forward times for each layer on client (seconds)
        client_backward_times: List of backward times for each layer on client (seconds)
        server_forward_times: List of forward times for each layer on server (seconds)
        server_backward_times: List of backward times for each layer on server (seconds)
        tensor_sizes: List of output tensor sizes (number of elements) for each layer
        network_throughput_mbps: Network throughput in Mbps

    Returns:
        optimal_split: Index of optimal split layer (0-based)
    """
    num_layers = len(client_fb_times)

    assert num_layers == len(server_fb_times)
    assert num_layers == len(tensor_sizes) # note: splitting after the last layer is not considered

    split_options = []

    # Evaluate each possible split point
    for split_layer in range(2, num_layers + 1):
        # split after `split_layer` layers
        # First layer has index 1
        # `split_layer=1` -> layer1 || layer2 layer3 ...
        # `split_layer=2` -> layer1 layer2 || layer3 ...

        # if `split_layer` == num_layers, it is local execution without any server offloading

        if split_layer == num_layers:
            # Local execution (no split) - all layers on client
            total_time = client_fb_times[-1]

        else:
            # Split execution

            # 1. Client computation time (layers 0 to split_layer)
            client_compute_time = client_fb_times[split_layer-1]

            # 2. Server computation time (from `split_layer` until the end)
            server_compute_time = sum(server_fb_times[split_layer:])

            # 3. Communication time
            # Forward: send intermediate activations from client to server
            forward_tensor_size_bits = tensor_sizes[split_layer - 1] * 32  # 4B per float32
            communication_time = 2 * forward_tensor_size_bits / network_throughput_Mbps / 1e6

            # Total training time for this split configuration
            total_time = client_compute_time + server_compute_time + communication_time

        split_options.append({
            "split_layer": split_layer,
            "total_time": total_time
        })

    # Find the split configuration with minimum training time
    optimal_config = min(split_options, key=lambda x: x["total_time"])

    return optimal_config["split_layer"], optimal_config["total_time"], split_options


# Example usage:
if __name__ == "__main__":
    # Example for a single device
    client_times = [
        {
            "forward_times": [0.01, 0.015, 10., 0.025, 0.03],
            "backward_times": [0.01, 0.015, 0.02, 0.025, 0.03],
        },
        {
            "forward_times": [0.01, 0.015, 0.02, 0.025, 0.03],
            "backward_times": [0.01, 0.015, 0.02, 0.025, 0.03],
        }
    ]
    server_times = {
        "forward_times": [0.01, 0.015, 0.02, 0.025, 0.03],
        "backward_times": [0.01, 0.015, 0.02, 0.025, 0.03],
    }
    tensor_sizes = [32*32*32, 32*16*16, 64*16*16, 64*8*8, 10]
    network_throughputs = [10, 20]  # Mbps

    split_vector, total_times = ares_compute_optimal_split_layer(
        client_times,
        server_times,
        tensor_sizes,
        network_throughputs
    )

    print(f"Optimal split layer: {split_vector}")
    print(f"Minimum training time: {total_times} seconds")
