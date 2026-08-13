from __future__ import annotations

import asyncio
import math
import struct
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import msgpack
import numpy as np

from .models import Endpoint


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024
DEFAULT_RPC_TIMEOUT = 30.0


class ProtocolError(RuntimeError):
    pass


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind == "f":
            array = array.astype("<f4", copy=False)
        elif array.dtype.kind in {"i", "u"}:
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)!r}")


def _msgpack_object_hook(value: Dict[str, Any]) -> Any:
    if value.get("__ndarray__") is True:
        try:
            dtype = np.dtype(str(value["dtype"]))
            raw_shape = value["shape"]
            raw = bytes(value["data"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed array payload") from exc
        if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
            raise ProtocolError(f"Unsupported array dtype: {dtype}")
        if dtype.byteorder == ">":
            raise ProtocolError("Array payloads must use little-endian byte order")
        if not isinstance(raw_shape, list) or len(raw_shape) > 8:
            raise ProtocolError("Malformed array dimensions")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in raw_shape):
            raise ProtocolError("Array dimensions must be integers")
        shape = tuple(int(item) for item in raw_shape)
        if any(item < 0 for item in shape):
            raise ProtocolError("Array dimensions cannot be negative")
        element_count = math.prod(shape)
        if element_count > MAX_FRAME_BYTES:
            raise ProtocolError("Array dimensions exceed the frame limit")
        expected = element_count * dtype.itemsize
        if expected != len(raw):
            raise ProtocolError("Array payload length does not match dtype and shape")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return value


def pack_message(message: Mapping[str, Any]) -> bytes:
    payload = msgpack.packb(dict(message), default=_msgpack_default, use_bin_type=True)
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"Message is larger than {MAX_FRAME_BYTES} bytes")
    return payload


def unpack_message(payload: bytes) -> Dict[str, Any]:
    try:
        value = msgpack.unpackb(payload, raw=False, object_hook=_msgpack_object_hook, strict_map_key=False)
    except (ValueError, TypeError, msgpack.ExtraData, msgpack.FormatError) as exc:
        raise ProtocolError("Invalid MessagePack payload") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Protocol message must be a map")
    return value


async def read_frame(reader: asyncio.StreamReader) -> tuple[Dict[str, Any], int]:
    header = await reader.readexactly(4)
    length = struct.unpack(">I", header)[0]
    if length <= 0 or length > MAX_FRAME_BYTES:
        raise ProtocolError(f"Invalid frame length: {length}")
    payload = await reader.readexactly(length)
    return unpack_message(payload), length + 4


async def write_frame(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> int:
    payload = pack_message(message)
    writer.write(struct.pack(">I", len(payload)))
    writer.write(payload)
    await writer.drain()
    return len(payload) + 4


def envelope(
    message_type: str,
    *,
    epoch: str,
    source_peer_id: int,
    source_endpoint: Endpoint,
    payload: Optional[Mapping[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": str(message_type),
        "message_id": uuid.uuid4().hex,
        "request_id": request_id,
        "epoch": str(epoch),
        "source_peer_id": int(source_peer_id),
        "source_endpoint": source_endpoint.to_wire(),
        "payload": dict(payload or {}),
    }


def validate_envelope(message: Mapping[str, Any], expected_epoch: Optional[str] = None) -> None:
    if int(message.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ProtocolError("Unsupported protocol version")
    for field in ("message_type", "message_id", "epoch", "source_peer_id", "source_endpoint", "payload"):
        if field not in message:
            raise ProtocolError(f"Missing envelope field: {field}")
    if expected_epoch is not None and str(message["epoch"]) != str(expected_epoch):
        raise ProtocolError("Message epoch does not match runtime epoch")


@dataclass
class RpcClient:
    source_peer_id: int
    source_endpoint: Endpoint
    epoch: str
    timeout: float = DEFAULT_RPC_TIMEOUT
    log: Optional[Callable[[str, Mapping[str, Any], Endpoint, int], None]] = None

    async def call(
        self,
        target: Endpoint,
        message_type: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        request = envelope(
            message_type,
            epoch=self.epoch,
            source_peer_id=self.source_peer_id,
            source_endpoint=self.source_endpoint,
            payload=payload,
        )

        async def exchange() -> Dict[str, Any]:
            reader, writer = await asyncio.open_connection(target.host, target.port)
            try:
                sent = await write_frame(writer, request)
                if self.log:
                    self.log("send", request, target, sent)
                response, received = await read_frame(reader)
                validate_envelope(response, self.epoch)
                if response.get("request_id") != request["message_id"]:
                    raise ProtocolError("RPC response request_id mismatch")
                if self.log:
                    self.log("receive", response, target, received)
                if response["message_type"] == "ERROR":
                    raise ProtocolError(str(response["payload"].get("error", "remote protocol error")))
                return response
            finally:
                writer.close()
                await writer.wait_closed()

        return await asyncio.wait_for(exchange(), timeout=self.timeout)
