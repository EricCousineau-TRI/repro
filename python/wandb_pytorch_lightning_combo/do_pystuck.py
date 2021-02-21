# begin: https://stackoverflow.com/a/45690594/7829525
import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
# end


_port = find_free_port()
print(f"""
    pystuck port: {_port}
""", flush=True)
import pystuck; pystuck.run_server(port=_port)
