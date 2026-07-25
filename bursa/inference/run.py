from bursa.inference.backend import BackendTransportError
from bursa.inference.constants import TRANSPORT_RETRY


def run_inference(backend, raw_prompt, grammar, n_predict, server=None) -> str:
    """Returns raw model content. Retries ONLY on transport failure (temp 0 makes content
    retries pointless). On transport error, health-check + restart the server if provided."""
    attempts = 0
    while True:
        try:
            return backend.generate(raw_prompt, grammar, n_predict)
        except BackendTransportError:
            attempts += 1
            if attempts > TRANSPORT_RETRY:
                raise
            if server is not None and not server.health():
                server.stop()
                server.start()
