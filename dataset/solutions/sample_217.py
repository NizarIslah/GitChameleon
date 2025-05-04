# library: mitmproxy
# version: 9.0.1
# extra_dependencies: []
import mitmproxy.connection as conn


def custom_server(ip_address: str, server_port: int) -> conn.Server:
    return conn.Server(address=(ip_address, server_port))
