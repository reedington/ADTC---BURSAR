from bursa.inference.server import build_server_args


def test_server_args_encode_d3_config():
    args = build_server_args("/m/model.gguf", port=8080, threads=4, ctx=2048)
    joined = " ".join(args)
    assert "llama-server" in joined
    assert "--model /m/model.gguf" in joined
    assert "--ctx-size 2048" in joined
    assert "--threads 4" in joined
    assert "--temp 0" in joined
    assert "--port 8080" in joined
    assert "q8_0" in joined
