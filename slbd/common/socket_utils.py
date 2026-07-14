"""Utilities for socket communication with message framing."""

import time
import pickle
import struct
import socket
import asyncio
from typing import Any


def enable_keepalive(sock: socket.socket) -> None:
    """Enable aggressive, permissive TCP keepalive on a socket.

    Sends keepalive probes after a short idle period so any NAT/conntrack/firewall
    state along the path (and the peer's liveness) is kept fresh during long idle
    gaps -- e.g. while a slow client runs its local forward pass. Tolerates a long
    burst of failed probes before ever declaring the peer dead, so a transient
    network stall does not tear the connection down.
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # start probing after a short idle period so middlebox state gets refreshed
    if hasattr(socket, "TCP_KEEPIDLE"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
    # probe frequently while idle
    if hasattr(socket, "TCP_KEEPINTVL"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
    # tolerate a very long run of failed probes before giving up (~8 min here)
    if hasattr(socket, "TCP_KEEPCNT"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 100)


def send_message(sock: socket.socket, data: Any) -> None:
    """Send a Python bytes over a socket with length prefix.

    Args:
        sock: Socket to send the message through
        serialized_data: bytes to send
    """
    serialized_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    length = len(serialized_data)
    # Send 4-byte length prefix followed by the data
    sock.sendall(struct.pack('!I', length))
    sock.sendall(serialized_data)


def recv_message(sock: socket.socket, node_name) -> Any:
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


async def _recv_exactly_async(sock: socket.socket, n: int, loop: asyncio.AbstractEventLoop) -> bytes:
    """Helper to receive exactly n bytes from non-blocking socket using asyncio.

    Args:
        sock: Non-blocking socket to receive from
        n: Number of bytes to receive
        loop: Event loop for async operations

    Returns:
        Received bytes or empty bytes if connection closed
    """
    data = bytearray(n)
    view = memoryview(data)
    received = 0
    message_start_time = time.perf_counter()
    while received < n:
        chunk = await loop.sock_recv(sock, n - received)
        if not chunk:
            return bytes(), None
        view[received:received+len(chunk)] = chunk
        received += len(chunk)
    rx_time = time.perf_counter() - message_start_time
    return bytes(data), rx_time


async def send_message_async(sock: socket.socket, data: Any, loop: asyncio.AbstractEventLoop) -> None:
    """Send a Python object over a non-blocking socket with length prefix using asyncio.

    Args:
        sock: Non-blocking socket to send the message through
        data: Python object to send
        loop: Event loop for async operations
    """
    serialized_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    length = len(serialized_data)
    # Send 4-byte length prefix followed by the data
    await loop.sock_sendall(sock, struct.pack('!I', length))
    await loop.sock_sendall(sock, serialized_data)


async def recv_message_async(sock: socket.socket, loop: asyncio.AbstractEventLoop) -> Any:
    """Receive a Python object from a non-blocking socket with length prefix using asyncio.

    Args:
        sock: Non-blocking socket to receive the message from
        loop: Event loop for async operations

    Returns:
        Tuple of (deserialized Python object, throughput in Mbps)
    """
    # Receive 4-byte length prefix
    length_data, _ = await _recv_exactly_async(sock, 4, loop)
    if not length_data:
        return None, None
    length = struct.unpack('!I', length_data)[0]

    # Receive the actual data
    data, rx_time = await _recv_exactly_async(sock, length, loop)
    if not data:
        return None, None

    throughput_Mbps = 8 * length / rx_time / 1e6
    return pickle.loads(data), throughput_Mbps
