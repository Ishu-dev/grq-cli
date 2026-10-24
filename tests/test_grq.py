import io, json, pathlib, urllib.error
import pytest
import grq


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def reply(text):
    return FakeResp(json.dumps({"choices": [{"message": {"content": text}}], "usage": {}}).encode())


def http_error(code, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "x", headers, io.BytesIO(b'{"error":{"message":"boom"}}'))


@pytest.fixture
def fake_urlopen(monkeypatch):
    calls = []
    script = []

    def urlopen(req, timeout):
        calls.append(json.loads(req.data) if req.data else None)
        step = script.pop(0) if script else reply("ok")
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(grq.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(grq.time, "sleep", lambda s: None)
    return calls, script


def test_retries_rate_limit_then_succeeds(fake_urlopen):
    calls, script = fake_urlopen
    script += [http_error(429, "1"), http_error(503), reply("fine")]
    assert grq.call("/chat/completions", "k", {"m": 1})["choices"][0]["message"]["content"] == "fine"
    assert len(calls) == 3


def test_client_error_is_not_retried(fake_urlopen):
    calls, script = fake_urlopen
    script.append(http_error(401))
    with pytest.raises(grq.GroqError, match="401: boom"):
        grq.call("/models", "k")
    assert len(calls) == 1


def test_batch_one_output_per_file_and_skips_done(tmp_path, fake_urlopen, monkeypatch):
    calls, _ = fake_urlopen
    src = tmp_path / "in"
    src.mkdir()
    for n in ("a", "b", "c"):
        (src / f"{n}.txt").write_text(n)
    out = tmp_path / "out"
    out.mkdir()
    (out / "b.md").write_text("old")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(grq.sys.stdin, "isatty", lambda: True, raising=False)

    grq.run(["classify", "--each", str(src / "*.txt"), "--outdir", str(out), "--ext", ".md"])

    assert (out / "a.md").read_text() == "ok\n"
    assert (out / "b.md").read_text() == "old"
    assert len(calls) == 2
    assert all("classify" in c["messages"][-1]["content"] for c in calls)


def test_each_requires_outdir(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(SystemExit):
        grq.run(["x", "--each", "*.txt"])
