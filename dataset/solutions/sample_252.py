# library: falcon
# version: 2.0.0
# extra_dependencies: []
from falcon.uri import parse_query_string


def custom_parse_query(qs: str) -> dict:
    return parse_query_string(qs, keep_blank=True, csv=False)
