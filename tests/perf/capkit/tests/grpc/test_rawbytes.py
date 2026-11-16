# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""U-1 (INTERFACES.md 12): a generic handler with identity (de)serializers and a client with
response_deserializer=None exchange raw bytes objects -- checked by type, in both directions."""

import asyncio

import grpc

from capgrpc.common import (AnyMethodHandler, bind_server, make_server, open_channel,
                                 raw_stream_callable)


def test_raw_bytes_both_directions():
    seen = {}

    def factory(method):
        async def behavior(request, context):
            seen["request_type"] = type(request)
            seen["method"] = method
            for i in range(3):
                yield b"\x00\x01" + bytes([i]) + request

        return behavior

    async def main():
        server = make_server(reuseport=False, max_unrequested_s=30)
        server.add_generic_rpc_handlers([AnyMethodHandler(factory)])
        port = bind_server(server, "127.0.0.1", 0)
        await server.start()
        ch = open_channel("127.0.0.1:%d" % port, local_subchannel_pool=True)
        try:
            call = raw_stream_callable(ch, "/any.Service/AnyMethod")(b"req-bytes")
            got = [m async for m in call]
            code = await call.code()
        finally:
            await ch.close()
            await server.stop(0)
        return got, code

    got, code = asyncio.run(main())
    assert code == grpc.StatusCode.OK
    assert seen["request_type"] is bytes and seen["method"] == "/any.Service/AnyMethod"
    assert [type(m) for m in got] == [bytes, bytes, bytes]
    assert got[2] == b"\x00\x01\x02req-bytes"
