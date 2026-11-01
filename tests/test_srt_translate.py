from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest

from srt_ollama_translate import ollama
from srt_ollama_translate.cli import main
from srt_ollama_translate.languages import language_from_tag
from srt_ollama_translate.ollama import OllamaSettings
from srt_ollama_translate.srt import read_srt, write_srt
from srt_ollama_translate.translator import plan_jobs, source_language_from_filename

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
Bonjour

2
00:00:03,000 --> 00:00:04,000
Comment ça va ?
deuxième ligne
"""


def test_language_tags():
    assert language_from_tag("fr").tag == "fr"
    assert language_from_tag("fra").tag == "fr"
    assert language_from_tag("pt-BR").tag == "pt-BR"
    assert language_from_tag("sr-Cyrl").tag == "sr-Cyrl"
    with pytest.raises(ValueError):
        language_from_tag("qsm")
    with pytest.raises(ValueError):
        language_from_tag("fr-ZZZZZ-XX")


@pytest.mark.parametrize("tag", ["zh", "la", "eo"])
def test_language_tags_accept_macro_historical_and_constructed(tag):
    assert language_from_tag(tag).tag == tag


@pytest.mark.parametrize("tag", ["und", "mul", "zxx"])
def test_language_tags_reject_special_codes(tag):
    with pytest.raises(ValueError, match="not a specific language"):
        language_from_tag(tag)


def test_source_language_from_filename(tmp_path):
    basename, source = source_language_from_filename(tmp_path / "my.show.fr.srt")
    assert basename == "my.show"
    assert source.tag == "fr"
    with pytest.raises(ValueError, match="Missing language"):
        source_language_from_filename(tmp_path / "episode.srt")
    with pytest.raises(ValueError, match="Invalid source language"):
        source_language_from_filename(tmp_path / "episode.qsm.srt")


def test_plan_jobs_rejects_same_language_twice(tmp_path):
    paths = [tmp_path / "ep.fr.srt", tmp_path / "ep.fra.srt"]
    with pytest.raises(ValueError, match="Several sources"):
        plan_jobs(paths, language_from_tag("en"), language_from_tag("fr"))


def test_plan_jobs_needs_from_for_several_sources(tmp_path):
    fr, en = tmp_path / "ep.fr.srt", tmp_path / "ep.en.srt"
    with pytest.raises(ValueError, match="pass --from"):
        plan_jobs([fr, en], language_from_tag("de"))
    jobs, notes = plan_jobs([fr, en], language_from_tag("de"), language_from_tag("fr"))
    assert [job.path for job in jobs] == [fr]
    assert notes == []


def test_plan_jobs_one_source_per_episode_needs_no_from(tmp_path):
    paths = [tmp_path / "a.fr.srt", tmp_path / "b.ja.srt", tmp_path / "b.en.srt"]
    jobs, _ = plan_jobs(paths, language_from_tag("en"))
    assert [job.path.name for job in jobs] == ["a.fr.srt", "b.ja.srt"]


def test_plan_jobs_from_skips_episodes_without_that_source(tmp_path):
    paths = [tmp_path / "a.fr.srt", tmp_path / "b.ja.srt"]
    jobs, notes = plan_jobs(paths, language_from_tag("en"), language_from_tag("fr"))
    assert [job.path.name for job in jobs] == ["a.fr.srt"]
    assert notes == ["Skipping b.en.srt: no French source"]


def test_plan_jobs_skips_finished_episode_before_ambiguity(tmp_path):
    (tmp_path / "ep.de.srt").write_text(SAMPLE, encoding="utf-8")
    jobs, notes = plan_jobs([tmp_path / "ep.fr.srt", tmp_path / "ep.en.srt"], language_from_tag("de"))
    assert jobs == []
    assert notes == ["Skipping ep.de.srt: already exists"]


def test_srt_round_trip(tmp_path):
    path = tmp_path / "ep.fr.srt"
    path.write_text(SAMPLE, encoding="utf-8")
    cues = read_srt(path)
    assert [cue.index for cue in cues] == ["1", "2"]
    assert cues[1].text == "Comment ça va ?\ndeuxième ligne"

    out = tmp_path / "ep.en.srt"
    write_srt(out, cues, ["Hello", "How are you?\nsecond line"])
    assert read_srt(out)[0] == cues[0]._replace(text="Hello")
    assert not list(tmp_path.glob("*.tmp"))


def test_read_srt_tolerates_extra_blank_lines_and_empty_cues(tmp_path):
    path = tmp_path / "ep.fr.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nA\n\n\n \n2\n00:00:03,000 --> 00:00:04,000\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nC\n",
        encoding="utf-8",
    )
    cues = read_srt(path)
    assert [cue.text for cue in cues] == ["A", "", "C"]

    write_srt(path, cues, [cue.text for cue in cues])
    assert read_srt(path) == cues


def test_read_srt_rejects_bad_block(tmp_path):
    path = tmp_path / "bad.fr.srt"
    path.write_text("not a subtitle\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SRT block"):
        read_srt(path)


class _FakeOllama(BaseHTTPRequestHandler):
    seen_requests: list[dict] = []
    reply: str | None = None
    done_reason = "stop"

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _FakeOllama.seen_requests.append(body)
        current = body["messages"][1]["content"].split("[CURRENT_SOURCE]\n", 1)[1]
        content = _FakeOllama.reply if _FakeOllama.reply is not None else f"EN:{current}"
        reply = json.dumps({"message": {"content": content}, "done_reason": _FakeOllama.done_reason}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_ollama():
    _FakeOllama.seen_requests = []
    _FakeOllama.reply = None
    _FakeOllama.done_reason = "stop"
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/api/chat"
    server.shutdown()
    server.server_close()


def test_end_to_end_with_fake_ollama(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.fr.srt").write_text(SAMPLE, encoding="utf-8")

    assert main(["--url", fake_ollama, "--max-cues", "1"]) == 0
    partial = tmp_path / "ep.en.srt.partial"
    assert [cue.text for cue in read_srt(partial)] == ["EN:Bonjour"]

    # Rerunning without a limit resumes from the checkpoint instead of re-translating cue 1.
    assert main(["--url", fake_ollama]) == 0
    output = tmp_path / "ep.en.srt"
    assert not partial.exists()
    assert [cue.text for cue in read_srt(output)] == [
        "EN:Bonjour",
        "EN:Comment ça va ?\ndeuxième ligne",
    ]
    assert len(_FakeOllama.seen_requests) == 2
    assert _FakeOllama.seen_requests[0]["options"] == {"temperature": 0, "num_predict": 256, "num_ctx": 4096}

    # Existing outputs are never overwritten.
    assert main(["--url", fake_ollama]) == 0
    assert len(_FakeOllama.seen_requests) == 2


def test_translate_cue_passes_style_and_model(fake_ollama):
    settings = OllamaSettings(model="m", url=fake_ollama, style="official")
    fr = language_from_tag("fr")
    en = language_from_tag("en")
    assert ollama.translate_cue("Salut", [], [], fr, en, settings) == "EN:Salut"
    body = _FakeOllama.seen_requests[-1]
    assert body["model"] == "m"
    assert "STYLE: official." in body["messages"][0]["content"]


def test_translate_cue_collapses_blank_lines(fake_ollama):
    _FakeOllama.reply = "Line one\n\n  \nLine two"
    fr, en = language_from_tag("fr"), language_from_tag("en")
    assert ollama.translate_cue("x", [], [], fr, en, OllamaSettings(url=fake_ollama)) == "Line one\nLine two"


def test_translate_cue_rejects_truncated_output(fake_ollama):
    _FakeOllama.done_reason = "length"
    fr, en = language_from_tag("fr"), language_from_tag("en")
    with pytest.raises(RuntimeError, match="truncated"):
        ollama.translate_cue("x", [], [], fr, en, OllamaSettings(url=fake_ollama))


def test_empty_cues_are_not_sent_to_model(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.fr.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nSalut\n", encoding="utf-8"
    )
    assert main(["--url", fake_ollama]) == 0
    assert [cue.text for cue in read_srt(tmp_path / "ep.en.srt")] == ["", "EN:Salut"]
    assert len(_FakeOllama.seen_requests) == 1


def test_bracketed_filename_is_taken_literally(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Show [1080p].fr.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["--url", fake_ollama, "Show [1080p].fr.srt"]) == 0
    assert (tmp_path / "Show [1080p].en.srt").exists()


def test_unmatched_explicit_argument_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.fr.srt").write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["ep.fr.srt", "typo.fr.srt"])


def test_from_picks_source_after_earlier_output(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.fr.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["--url", fake_ollama]) == 0
    with pytest.raises(SystemExit):
        main(["--url", fake_ollama, "--to", "de"])
    assert main(["--url", fake_ollama, "--to", "de", "--from", "fr"]) == 0
    assert (tmp_path / "ep.de.srt").exists()


def test_resume_refuses_different_style(tmp_path, monkeypatch, fake_ollama):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.fr.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["--url", fake_ollama, "--max-cues", "1"]) == 0
    assert main(["--url", fake_ollama, "--style", "official"]) == 1
    assert main(["--url", fake_ollama]) == 0
    assert not (tmp_path / "ep.en.srt.partial.json").exists()
