# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Python gRPC transport over real sockets without running a GPU pipeline."""

import asyncio
import datetime
import importlib
import ipaddress
import shutil
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("grpc_health")
pytest.importorskip("holoscan.core")
pytest.importorskip("cupy")
pytest.importorskip("cryptography")
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from grpc_health.v1 import health_pb2, health_pb2_grpc


@pytest.fixture(scope="module")
def rpc(tmp_path_factory):
    protoc = pytest.importorskip("grpc_tools.protoc")
    generated = tmp_path_factory.mktemp("grpc_protos")
    proto = generated / "holohub/grpc_operators/holoscan.proto"
    proto.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[2] / "protos/holoscan.proto", proto)
    assert (
        protoc.main(
            [
                "protoc",
                f"-I{generated}",
                f"--python_out={generated}",
                f"--grpc_python_out={generated}",
                str(proto),
            ]
        )
        == 0
    )
    sys.path.insert(0, str(generated))
    try:
        yield SimpleNamespace(
            messages=importlib.import_module("holohub.grpc_operators.holoscan_pb2"),
            stubs=importlib.import_module("holohub.grpc_operators.holoscan_pb2_grpc"),
            service=importlib.import_module(
                "operators.grpc_operators.python.server.grpc_service"
            ).GrpcService,
            servicer=importlib.import_module(
                "operators.grpc_operators.python.server.entity_servicer"
            ).HoloscanEntityServicer,
            client=importlib.import_module(
                "operators.grpc_operators.python.client.entity_client_service"
            ).EntityClientService,
        )
    finally:
        sys.path.remove(str(generated))


@pytest.fixture(scope="module")
def remote_address():
    # UDP connect selects a local interface without sending packets to this TEST-NET address.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
        except OSError:
            pytest.skip("A non-loopback IPv4 interface is required")
        return probe.getsockname()[0]


@pytest.fixture(scope="module")
def certificates(remote_address):
    def issue(name, issuer=None, ca=False):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = datetime.datetime.now(datetime.timezone.utc)
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer[1].subject if issuer else subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        )
        if name == "server":
            builder = builder.add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName("localhost"),
                        x509.IPAddress(ipaddress.ip_address(remote_address)),
                        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    ]
                ),
                critical=False,
            )
        return key, builder.sign(issuer[0] if issuer else key, hashes.SHA256())

    def pem(pair):
        key, cert = pair
        return (
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            cert.public_bytes(serialization.Encoding.PEM),
        )

    ca = issue("trusted CA", ca=True)
    other_ca = issue("untrusted CA", ca=True)
    return SimpleNamespace(
        ca=pem(ca)[1],
        other_ca=pem(other_ca)[1],
        server=pem(issue("server", ca)),
        client=pem(issue("client", ca)),
        untrusted=pem(issue("client", other_ca)),
    )


@asynccontextmanager
async def running_service(rpc, host="127.0.0.1", reinitialize=False, **tls):
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as reserve:
        reserve.bind(("::1" if ":" in host else "127.0.0.1", 0))
        port = reserve.getsockname()[1]
    factory = SimpleNamespace(created=[])

    def create(name, incoming, outgoing):
        app = SimpleNamespace(
            composed=True,
            enqueue_request=lambda request: outgoing.put(
                rpc.messages.EntityResponse(parameters=request.parameters)
            ),
            enqueue_response=outgoing.put,
            is_response_available=lambda: not outgoing.empty(),
            dequeue_response=outgoing.get,
        )
        factory.created.append(app)
        return app

    factory.create_new_application_instance = create
    factory.destroy_application_instance = lambda app: None
    service = rpc.service()
    # Use the default call unchanged for the pre-fix remote-access reproduction.
    kwargs = {} if host == "127.0.0.1" and not tls else {"host": host, **tls}
    service.initialize(port, factory, **kwargs)
    if reinitialize:
        with pytest.raises(ValueError):
            service.initialize(port, factory, host=host, certificate_chain=tls["certificate_chain"])
    task = asyncio.create_task(service.start([rpc.servicer("test")]))
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host

    async def wait_for_server():
        while True:
            if task.done():
                await task
            try:
                _, writer = await asyncio.open_connection(connect_host, port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError:
                await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(wait_for_server(), timeout=5)
        yield port, factory
    finally:
        await service.stop()
        await task


async def entity_probe(rpc, channel):
    responses = [
        response
        async for response in rpc.stubs.EntityStub(channel).EntityStream(
            iter([rpc.messages.EntityRequest(end_of_stream=True)]), timeout=2
        )
    ]
    assert len(responses) == 1 and responses[0].end_of_stream


def test_default_service_rejects_non_loopback_client(rpc, remote_address):
    async def run():
        async with running_service(rpc) as (port, factory):
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                await entity_probe(rpc, channel)
            assert len(factory.created) == 1
            async with grpc.aio.insecure_channel(f"{remote_address}:{port}") as channel:
                with pytest.raises(grpc.aio.AioRpcError):
                    await entity_probe(rpc, channel)
            assert len(factory.created) == 1

    asyncio.run(run())


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "localhost.example"])
def test_remote_bind_requires_mtls(rpc, host):
    with pytest.raises(ValueError):
        rpc.service().initialize(50051, None, host=host)


@pytest.mark.parametrize(
    "target",
    [
        "0.0.0.0:50051",
        "192.0.2.1:50051",
        "[::]:50051",
        "localhost.example:50051",
        "dns:///localhost:50051",
    ],
)
def test_client_rejects_remote_plaintext(rpc, target):
    with pytest.raises(ValueError):
        rpc.client(target, False, None, None, None)


@pytest.mark.parametrize(
    "missing", ["private_key", "certificate_chain", "root_certificates", "all_empty"]
)
def test_partial_server_credentials_fail_closed(rpc, certificates, missing):
    credentials = {
        "private_key": certificates.server[0],
        "certificate_chain": certificates.server[1],
        "root_certificates": certificates.ca,
    }
    if missing == "all_empty":
        credentials = dict.fromkeys(credentials, b"")
    else:
        credentials[missing] = None
    with pytest.raises(ValueError):
        rpc.service().initialize(50051, None, **credentials)


@pytest.mark.parametrize(
    "client_kind",
    [
        "plaintext",
        "no_certificate",
        "untrusted_client",
        "untrusted_server",
        "plaintext_after_bad_reinitialization",
    ],
)
def test_mtls_rejects_unauthorized_clients(rpc, certificates, remote_address, client_kind):
    async def run():
        async with running_service(
            rpc,
            host="0.0.0.0",
            private_key=certificates.server[0],
            certificate_chain=certificates.server[1],
            root_certificates=certificates.ca,
            reinitialize=client_kind == "plaintext_after_bad_reinitialization",
        ) as (port, factory):
            target = f"{remote_address}:{port}"
            key, cert = (
                certificates.untrusted if client_kind == "untrusted_client" else certificates.client
            )
            ca = certificates.other_ca if client_kind == "untrusted_server" else certificates.ca
            credentials = grpc.ssl_channel_credentials(
                ca,
                None if client_kind == "no_certificate" else key,
                None if client_kind == "no_certificate" else cert,
            )
            channel = (
                grpc.aio.insecure_channel(target)
                if client_kind.startswith("plaintext")
                else grpc.aio.secure_channel(target, credentials)
            )
            async with channel:
                with pytest.raises(grpc.aio.AioRpcError):
                    await entity_probe(rpc, channel)
                with pytest.raises(grpc.aio.AioRpcError):
                    await health_pb2_grpc.HealthStub(channel).Check(
                        health_pb2.HealthCheckRequest(), timeout=2
                    )
            assert not factory.created

    asyncio.run(run())


def test_invalid_server_certificates_do_not_fall_back_to_plaintext(rpc):
    async def run():
        with pytest.raises(RuntimeError):
            async with running_service(
                rpc,
                host="0.0.0.0",
                private_key=b"invalid",
                certificate_chain=b"invalid",
                root_certificates=b"invalid",
            ):
                pytest.fail("Invalid TLS credentials must prevent server startup")

    asyncio.run(run())


@pytest.fixture(params=["cloud", "edge"])
def command(request, rpc, monkeypatch):
    app_dir = (
        Path(__file__).parents[4]
        / "applications/distributed/grpc/grpc_endoscopy_tool_tracking/python"
    )
    monkeypatch.syspath_prepend(str(app_dir / request.param))
    # The CLI does not instantiate the GPU inference pipeline while parsing arguments.
    monkeypatch.setitem(
        sys.modules,
        "endoscopy_tool_tracking",
        SimpleNamespace(EndoscopyToolTrackingPipeline=object),
    )
    spec = importlib.util.spec_from_file_location(
        f"grpc_{request.param}_command", app_dir / request.param / f"app_{request.param}_main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_accepts_complete_tls_configuration(command, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["demo", "--tls-ca", "ca.pem", "--tls-cert", "cert.pem", "--tls-key", "key.pem"],
    )
    args = command.parse_arguments()
    assert (args.tls_ca, args.tls_cert, args.tls_key) == (
        Path("ca.pem"),
        Path("cert.pem"),
        Path("key.pem"),
    )


@pytest.mark.parametrize("flag", ["--tls-ca", "--tls-cert", "--tls-key"])
def test_demo_rejects_partial_tls_configuration(command, monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["demo", flag, "cert.pem"])
    with pytest.raises(SystemExit) as error:
        command.parse_arguments()
    assert error.value.code == 2


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "0.0.0.0"])
def test_entity_client_streams_with_permitted_transport(rpc, certificates, remote_address, host):
    async def run():
        tls = (
            {
                "private_key": certificates.server[0],
                "certificate_chain": certificates.server[1],
                "root_certificates": certificates.ca,
            }
            if host == "0.0.0.0"
            else {}
        )
        credentials = (
            grpc.ssl_channel_credentials(certificates.ca, *certificates.client) if tls else None
        )
        async with running_service(rpc, host=host, **tls) as (port, factory):
            target = (
                f"{remote_address}:{port}"
                if tls
                else (f"[::1]:{port}" if host == "::1" else f"localhost:{port}")
            )
            outgoing, incoming = asyncio.Queue(), Queue()
            await outgoing.put(rpc.messages.EntityRequest(parameters={"probe": "authorized"}))
            await outgoing.put(rpc.messages.EntityRequest(end_of_stream=True))
            client = rpc.client(
                target,
                False,
                SimpleNamespace(empty=outgoing.empty, pop=outgoing.get),
                SimpleNamespace(push=incoming.put),
                None,
                credentials=credentials,
            )
            await asyncio.wait_for(client.start_entity_stream(), timeout=5)
            assert incoming.get_nowait().parameters["probe"] == "authorized"
            assert incoming.get_nowait().end_of_stream
            assert len(factory.created) == 1

    asyncio.run(run())


@pytest.mark.parametrize("proxy_type", ["name", "address"])
def test_plaintext_loopback_client_does_not_use_http_proxy(rpc, monkeypatch, proxy_type):
    async def run():
        proxied = asyncio.Event()

        async def proxy_connection(reader, writer):
            proxied.set()
            writer.close()
            await writer.wait_closed()

        async with await asyncio.start_server(proxy_connection, "127.0.0.1", 0) as proxy:
            proxy_port = proxy.sockets[0].getsockname()[1]
            if proxy_type == "name":
                monkeypatch.setenv("grpc_proxy", f"http://127.0.0.1:{proxy_port}")
            else:
                monkeypatch.setenv("GRPC_ADDRESS_HTTP_PROXY", f"127.0.0.1:{proxy_port}")
                monkeypatch.setenv("GRPC_ADDRESS_HTTP_PROXY_ENABLED_ADDRESSES", "127.0.0.0/8")
            monkeypatch.setenv("no_grpc_proxy", "")
            async with running_service(rpc) as (port, factory):
                outgoing, incoming = asyncio.Queue(), Queue()
                await outgoing.put(rpc.messages.EntityRequest(end_of_stream=True))
                client = rpc.client(
                    f"localhost:{port}",
                    False,
                    SimpleNamespace(empty=outgoing.empty, pop=outgoing.get),
                    SimpleNamespace(push=incoming.put),
                    None,
                )
                stream = asyncio.create_task(client.start_entity_stream())
                proxy_used = asyncio.create_task(proxied.wait())
                try:
                    await asyncio.wait(
                        [stream, proxy_used], timeout=3, return_when=asyncio.FIRST_COMPLETED
                    )
                    assert not proxied.is_set(), "Plaintext loopback traffic reached an HTTP proxy"
                    assert stream.done(), "The local client did not complete"
                    await stream
                    assert incoming.get_nowait().end_of_stream
                    assert len(factory.created) == 1
                finally:
                    stream.cancel()
                    proxy_used.cancel()
                    await asyncio.gather(stream, proxy_used, return_exceptions=True)

    asyncio.run(run())
