"""Utilities for socket communication with message framing."""

import pickle
import struct
import socket
from typing import Any


def send_message(sock: socket.socket, serialized_data: bytes) -> None:
    """Send a Python bytes over a socket with length prefix.

    Args:
        sock: Socket to send the message through
        serialized_data: bytes to send
    """
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
    length_data = _recv_exactly(sock, 4)
    if not length_data:
        return None
    length = struct.unpack('!I', length_data)[0]

    # Receive the actual data
    data = _recv_exactly(sock, length)
    if not data:
        return None

    return pickle.loads(data)


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
    while received < n:
        chunk = sock.recv(n - received)
        if not chunk:
            return bytes()
        view[received:received+len(chunk)] = chunk
        received += len(chunk)
    return bytes(data)
