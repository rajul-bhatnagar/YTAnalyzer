"""Unit tests for Phase 3 pure logic: schemas, validation, audio/visual signals, fusion, scoring, clips, providers."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from google.genai import errors as genai_errors

from pipeline import audio_analysis, event_detection, frame_sampler, gameplay_analysis as ga, highlight_scoring as hs
from pipeline.ai_providers.base import (ProviderError, ProviderUnavailable, QuotaExceeded, WindowAnalysis,
                                        flat_schema)
from pipeline.ai_providers.gemini_provider import GeminiProvider, _find_retry_delay
from pipeline.ai_providers.ollama_provider import OllamaProvider


# --- schema & windows -------------------------------------------------------------------------------------
def test_flat_schema_is_self_contained():
    s = json.dumps(flat_schema(WindowAnalysis))
    assert "$ref" not in s and "$defs" not in s and '"null"' not in s and '"title"' not in s
    assert '"significance"' in s and '"minimum": 0' in s and '"maximum": 5' in s


def test_build_windows_merges_short_tail():
    w = ga.build_windows(380, 180)
    assert [(x.start, x.end) for x in w] == [(0, 180), (180, 380)]  # 20 s tail (< 30 s) merged
    assert [(x.start, x.end) for x in ga.build_windows(400, 180)] == [(0, 180), (180, 360), (360, 400)]
    assert ga.build_windows(20, 180)[0].end == 20


def test_budget_stops_at_limit():
    b = ga.Budget(2, 0.0)
    assert b.take(None) and b.take(None) and not b.take(None)


# --- validation ------------------------------------------------------------------------------------------
def _ev(**kw):
    e = {"start_sec": 10, "end_sec": 12, "category": "Clutch", "description": "d", "frame_times": [10.0],
         "transcript_quote": "", "confidence": {"visual": "high", "audio": "none", "transcript": "none"}}
    e.update(kw)
    return e


def _evidence(transcript="what a catch"):
    return ga.WindowEvidence(ga.Window(0, 0.0, 60.0), [], [], [{"start": 9, "end": 11, "text": transcript}],
                             {"game": "G", "style": "Custom", "topic": "", "language": "en"})


def _raw(*events):
    return json.dumps({"summary": "", "events": list(events)})


def test_validate_keeps_good_and_fixes_bad_references():
    wa, issues = ga.validate_result(_raw(_ev(frame_times=[10.2, 33.3], transcript_quote="WHAT a catch!")),
                                    _evidence(), "images", [10.0, 20.0], 60)
    e = wa.events[0]
    assert e.frame_times == [10.0]  # snapped to the shown frame; unseen 33.3 removed
    assert e.transcript_quote == "WHAT a catch!"  # punctuation/case-insensitive match is accepted
    assert "not shown" in e.uncertainty and e.category == "clutch"


def test_validate_drops_out_of_window_and_reverses_times():
    wa, issues = ga.validate_result(_raw(_ev(start_sec=500, end_sec=510), _ev(start_sec=20, end_sec=15)),
                                    _evidence(), "images", [10.0], 60)
    assert len(wa.events) == 1 and (wa.events[0].start_sec, wa.events[0].end_sec) == (15, 20)
    assert any("outside" in i for i in issues) and any("Swapped" in i for i in issues)


def test_validate_text_mode_strips_visual_claims_and_fake_quotes():
    wa, _ = ga.validate_result(_raw(_ev(visual_observation="I see a goal", transcript_quote="we won the cup",
                                        confidence={"visual": "high", "audio": "low", "transcript": "high"})),
                               _evidence(), "text", [], 60)
    e = wa.events[0]
    assert e.frame_times == [] and e.visual_observation == "" and e.confidence.visual == "none"
    assert e.transcript_quote == "" and e.confidence.transcript == "low"


def test_validate_images_without_cited_frame_lowers_visual_confidence():
    wa, _ = ga.validate_result(_raw(_ev(frame_times=[])), _evidence(), "images", [10.0], 60)
    assert wa.events[0].confidence.visual == "low"


def test_validate_video_mode_converts_clip_relative_times():
    ev = _evidence()
    ev.window = ga.Window(1, 100.0, 160.0)
    wa, _ = ga.validate_result(_raw(_ev(start_sec=5, end_sec=8, frame_times=[6])), ev, "video", [], 300)
    assert (wa.events[0].start_sec, wa.events[0].end_sec, wa.events[0].frame_times) == (105, 108, [106])


@pytest.mark.parametrize("raw", ["{not json", json.dumps({"events": [{"start_sec": "x"}]}),
                                 json.dumps({"events": [_ev(factors={"skill": 9})]})])
def test_validate_rejects_malformed_output(raw):
    with pytest.raises(ga.InvalidOutput):
        ga.validate_result(raw, _evidence(), "images", [10.0], 60)


# --- audio signals --------------------------------------------------------------------------------------
def test_game_events_sudden_vs_gradual():
    db = np.full(600, -40.0, np.float32)
    db[200:210] = -10.0  # sudden 1 s burst (flat; maximum may be anywhere inside)
    db[209] = -9.0
    ramp = np.linspace(-40, -12, 20)
    db[400:420] = ramp
    db[420:430] = -12.0
    ev = audio_analysis.game_audio_events(db, 0.1, 10.0)
    kinds = {round(e["start"]): e["kind"] for e in ev}
    assert kinds[20] == "game_transient"
    assert any(k == "loudness_rise" for s, k in kinds.items() if 40 <= s <= 42)


def test_voice_emphasis_needs_a_real_baseline():
    db = np.full(1000, -30.0, np.float32)
    one = [(10.0, 18.0)]
    db[100:180] = np.linspace(-35, -20, 80)  # natural variation inside one segment
    ev, meta = audio_analysis.voice_emphasis_events(db, 0.1, one, 6.0)
    assert ev == [] and "Not enough speech" in meta["skipped"]
    segs = [(0.0, 9.0), (20.0, 29.0), (40.0, 49.0), (60.0, 69.0)]
    db2 = np.full(1000, -60.0, np.float32)
    for s, e in segs:
        db2[int(s * 10):int(e * 10)] = -30.0
    db2[600:690] = -18.0  # the last segment is clearly louder
    ev2, _ = audio_analysis.voice_emphasis_events(db2, 0.1, segs, 6.0)
    assert [round(e["start"]) for e in ev2] == [60] and ev2[0]["strength_db"] == pytest.approx(12, abs=0.5)


def test_clipping_ranges():
    counts = np.zeros(100, np.int32)
    counts[10:13] = 5
    assert audio_analysis.clipping_ranges(counts, 0.1) == [{"start": 1.0, "end": 1.3, "samples": 15}]


# --- visual signals & frames ----------------------------------------------------------------------------
def test_scene_changes_find_hard_cuts_only():
    grid = np.arange(0, 60, 0.5)
    motion = np.full(len(grid), 2.0) + np.random.default_rng(0).normal(0, 0.2, len(grid))
    motion[40] = 60.0
    out = frame_sampler.scene_changes(grid, motion, 6.0, 12.0)
    assert [s["peak"] for s in out] == [20.0]


def test_choose_timestamps_prioritises_evidence_and_caps():
    t = frame_sampler.choose_timestamps(0, 60, 5, [30.0], 2, 6, 60)
    times = [x for x, _ in t]
    assert len(t) == 6 and 30.0 in times and 29.0 in times
    assert all(0 <= x < 60 for x in times)
    assert all(b - a >= 0.75 for a, b in zip(times, times[1:]))
    # Near the end of the recording frames never go past the last decodable moment.
    t2 = frame_sampler.choose_timestamps(50, 60, 5, [59.9], 2, 10, 60)
    assert max(x for x, _ in t2) <= 59.95


# --- clustering ----------------------------------------------------------------------------------------------
def _atom(kind, s, e, peak, **kw):
    return {"kind": kind, "start": s, "end": e, "peak": peak, "note": kind, **kw}


def test_cluster_combines_modalities_and_flags_single_signal():
    atoms = [_atom("game_transient", 40, 41, 40.5, strength_db=24), _atom("scene_change", 40, 40.5, 40, strength=3),
             _atom("scene_change", 80, 80.5, 80, strength=3)]
    words = [{"start": 41.0, "end": 41.4, "word": " What"}, {"start": 41.4, "end": 41.8, "word": " a shot"}]
    ev = event_detection.cluster(atoms, words, 3.0, 100.0)
    assert len(ev) == 2
    big, small = ev
    assert big["modalities"] == ["audio", "visual"] and big["strength"] > small["strength"]
    assert big["transcript_quote"] == "What a shot" and big["uncertainty"] == ""
    assert "Picture change only" in small["uncertainty"]


# --- scoring & candidates ------------------------------------------------------------------------------------
def _sig_event(start=40, end=41, strength=0.9):
    return {"key": f"s_{start}", "start": start, "end": end, "peak": start + 0.5, "kinds": ["game_transient"],
            "modalities": ["audio", "visual"], "strength": strength, "description": "Measured: x.",
            "transcript_quote": "", "uncertainty": "", "clipping": False,
            "members": [_atom("game_transient", start, end, start + 0.5, strength_db=20)]}


def test_zero_ai_ratings_are_excluded_and_signals_only_is_capped():
    ev = {"start": 40, "end": 41, "peak": 40.5, "signals": [_sig_event()], "ai": None}
    f = hs.factor_scores(ev, True, False, [])
    score, b = hs.score_event(f, "Custom", True)
    assert score <= hs.SIGNALS_ONLY_CAP and "significance" in b["excluded"]
    ai = {"factors": {k: 0 for k in hs.AI_FACTORS} | {"significance": 5}, "confidence": {}, "frame_times": [],
          "transcript_quote": ""}
    f2 = hs.factor_scores({**ev, "ai": ai}, True, False, [])
    assert f2["significance"] == 1.0 and f2["skill"] is None  # 0 = cannot judge -> excluded


def test_style_changes_weights_not_evidence():
    w_sport, w_horror = hs.weights_for("Sports Clutch / Match"), hs.weights_for("Horror / Funny Reactions")
    assert w_sport["stakes"] > w_horror["stakes"] and w_horror["commentary_support"] > w_sport["commentary_support"]
    assert hs.band(80) == "strong" and hs.band(56) == "promising" and hs.band(10) == "weak"


def _signals(duration=120.0, speech=None):
    grid = np.arange(0, duration, 0.5, dtype=np.float32)
    motion = np.full(len(grid), 5.0, np.float32)
    return {"grid": grid, "motion": motion, "speech": speech, "meta": {"hop_sec": 0.5, "has_game_audio": True,
                                                                       "has_speech_track": speech is not None}}


CFG = {"min_sec": 20, "max_sec": 50, "context_before_sec": 8, "aftermath_sec": 6, "handle_sec": 5,
       "candidate_count": 3, "allow_overlap": False}


@pytest.mark.parametrize("start,end", [(1.0, 2.0), (115.0, 119.5), (50.0, 51.0), (30.0, 110.0)])
def test_build_clip_bounds_and_lengths(start, end):
    sig = _signals()
    c = hs.build_clip({"start": start, "end": end, "peak": start + 0.5}, 120.0, CFG, sig["grid"], sig["motion"], [])
    assert 0 <= c["start_sec"] < c["end_sec"] <= 120.0
    assert 20 - 0.01 <= c["end_sec"] - c["start_sec"] <= 50 + 0.01
    assert c["start_sec"] <= c["event_sec"] <= c["end_sec"]
    assert c["handles"]["before"] <= c["start_sec"] and c["handles"]["after"] <= 120.0 - c["end_sec"]


def test_build_clip_never_cuts_mid_sentence():
    sig = _signals()
    runs = [(28.0, 34.0)]  # speech around the natural start (40 - 8 - padding)
    c = hs.build_clip({"start": 40.0, "end": 41.0, "peak": 40.5}, 120.0, CFG, sig["grid"], sig["motion"], runs)
    assert not (28.0 < c["start_sec"] < 34.0)


def test_build_clip_short_recording():
    sig = _signals(12.0)
    c = hs.build_clip({"start": 5.0, "end": 6.0, "peak": 5.5}, 12.0, CFG, sig["grid"], sig["motion"], [])
    assert (c["start_sec"], c["end_sec"]) == (0.0, 12.0) and "shorter than the minimum" in " ".join(c["clip_notes"])


def test_candidates_dedupe_and_respect_blocked_ranges():
    fused = hs.fuse([_sig_event(40, 41, 0.95), _sig_event(43, 44, 0.9), _sig_event(100, 101, 0.5)], {})
    cands = hs.build_candidates(fused, 150.0, "Custom", CFG, _signals(150.0), [], count=3)
    spans = [(c["start_sec"], c["end_sec"]) for c in cands]
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1:]:
            assert a1 <= b0 or b1 <= a0
    assert any(len(c["event_keys"]) > 1 for c in cands)  # the 43 s event merged into the 40 s clip
    blocked = hs.build_candidates(fused, 150.0, "Custom", CFG, _signals(150.0), [], count=3, blocked=[(20, 70)])
    assert all(c["start_sec"] >= 70 or c["end_sec"] <= 20 for c in blocked)


# --- providers -----------------------------------------------------------------------------------------------
def test_gemini_error_mapping():
    g = GeminiProvider("k", "m")
    q = g._map_error(genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                               "details": [{"retryDelay": "17s"}]}}))
    assert isinstance(q, QuotaExceeded) and q.retry_after_sec == 17.0
    assert isinstance(g._map_error(genai_errors.ClientError(404, {"error": {"code": 404}})), ProviderUnavailable)
    assert isinstance(g._map_error(genai_errors.ClientError(403, {"error": {"code": 403}})), ProviderUnavailable)
    s = g._map_error(genai_errors.ServerError(503, {"error": {"code": 503}}))
    assert type(s) is ProviderError and s.transient
    assert _find_retry_delay({"a": [{"b": {"retryDelay": "2.5s"}}]}) == 2.5


def test_gemini_without_key_is_unavailable_and_repr_hides_key():
    g = GeminiProvider("", "gemini-3.8-flash")
    assert not g.status().ok and "GEMINI_API_KEY" in g.status().message
    assert "secret" not in repr(GeminiProvider("secret", "m"))


def _ollama(handler):
    return OllamaProvider("http://ollama.test", "m:1b", transport=httpx.MockTransport(handler))


def test_ollama_errors_are_mapped():
    assert isinstance(pytest.raises(ProviderUnavailable, _ollama(
        lambda r: httpx.Response(404, json={"error": "model not found"})).generate_json, "s", "p", {}).value,
        ProviderUnavailable)
    err = pytest.raises(ProviderError, _ollama(lambda r: httpx.Response(200, json={
        "message": {"content": "{"}, "done_reason": "length"})).generate_json, "s", "p", {}).value
    assert "length limit" in str(err)
    err = pytest.raises(ProviderError, _ollama(lambda r: httpx.Response(500, text="boom")).generate_json,
                        "s", "p", {}).value
    assert err.transient

    def slow(r):
        raise httpx.ReadTimeout("slow", request=r)
    err = pytest.raises(ProviderError, _ollama(slow).generate_json, "s", "p", {}).value
    assert err.transient and "did not answer" in str(err)


def test_ollama_text_model_refuses_images_and_never_pulls():
    seen = []

    def handler(r):
        seen.append(r.url.path)
        if r.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(200, json={"message": {"content": "{}"}, "done_reason": "stop"})
    p = _ollama(handler)
    from pipeline.ai_providers.base import ImageInput
    with pytest.raises(ProviderUnavailable):
        p.generate_json("s", "p", {}, images=[ImageInput(b"x", "f")])
    assert p.status().ok and "text-only" in p.status().message
    assert "/api/pull" not in seen
