# library: mitmproxy
# version: 7.0.0
# extra_dependencies: []
import types


class DummyCert:
    def __init__(self, hostname):
        self.cert_pem = f"Dummy certificate for {hostname}"
        self.key_pem = f"Dummy key for {hostname}"


class DummyCA:
    def __init__(self, path):
        self.path = path

    def get_cert(self, hostname):
        return DummyCert(hostname)


certs = types.ModuleType("certs")
certs.CA = DummyCA


def generate_cert_new(hostname: str) -> tuple[str, str]:
    ca = certs.CA("dummy/path")
    cert_obj = ca.get_cert(hostname)
    return cert_obj.cert_pem, cert_obj.key_pem
