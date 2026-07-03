from typing import Dict, List


def compute_server_time(
    candidate_split_vector: Dict[str, int],
    pending_batches: Dict[str, int],
    server_training_times: List[float],
    per_batch_server_communication_time: List[float],
    num_concat_batches: int
):
    """ Computes the remaining epoch time for the server if clients train the number of layers as specified
    in the candidate_split_vector, which contains for every `cid` (key) a value stating how many layers it trains
    """
    server_time = 0. # Compute the time the server needs to train with the current configuration
    server_communication_time = 0. # time the server is communicating

    for cid, L in candidate_split_vector.items():
        B = pending_batches[cid] # number of batches the client still has to process
        server_time += B * server_training_times[L] / num_concat_batches
        server_communication_time += B * per_batch_server_communication_time[L]

    server_time = max(server_time, server_communication_time)
    return server_time
