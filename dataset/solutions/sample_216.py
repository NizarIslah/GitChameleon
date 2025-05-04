# library: mitmproxy
# version: 9.0.1
# extra_dependencies: []
import time
import mitmproxy.connection as conn


def custom_client(ip_address: str, i_port: int, o_port: int) -> conn.Client:
    return conn.Client(
        peername=(ip_address, i_port),
        sockname=(ip_address, o_port),
        timestamp_start=time.time(),
    )
