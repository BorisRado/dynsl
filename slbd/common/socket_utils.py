"""Utilities for socket communication with message framing."""

import time
import pickle
import struct
import socket
from typing import Any


def send_message(sock: socket.socket, data: Any) -> None:
    """Send a Python bytes over a socket with length prefix.

    Args:
        sock: Socket to send the message through
        serialized_data: bytes to send
    """
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    # Send 4-byte length prefix followed by the data
    sock.sendall(struct.pack('!I', length))
    sock.sendall(serialized_data)


def recv_message(sock: socket.socket) -> Any:
    """Receive a Python object from a socket with length prefix.

    Args:
        sock: Socket to receive the message from

    Returns:
        Deserialized Python object
    """
    # Receive 4-byte length prefix
    length_data, _ = _recv_exactly(sock, 4)
    if not length_data:
        return None, None
    length = struct.unpack('!I', length_data)[0]

    # Receive the actual data
    data, rx_time = _recv_exactly(sock, length)
    if not data:
        return None, None

    throughput_Mbps = 8 * length / rx_time / 1e6
    return pickle.loads(data), throughput_Mbps


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Helper to receive exactly n bytes from socket.

    Args:
        sock: Socket to receive from
        n: Number of bytes to receive

    Returns:
        Received bytes or empty bytes if connection closed
    """
    data = bytearray(n)
    view = memoryview(data)
    received = 0
    # if _recv_exactly is called for the length -> we always ignore the rx_time
    # _recv_exactly is otherwise called immediately after the length, so we should start measuring *before*
    # receiving the first chunk of data
    message_start_time = time.perf_counter()
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            return bytes(), None
        view[received:received+len(chunk)] = chunk
        received += len(chunk)
    rx_time = time.perf_counter() - message_start_time
    return bytes(data), rx_time
