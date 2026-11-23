"""
Tests for the LLM layer. The Gemini tests swap in a tiny fake of Google's
Gen AI SDK, so they check how GeminiLLM calls the SDK without a network,
a key, or the SDK installed. The retry tests use a fake clock, so the
waits they check take no real time.

RUN:  pytest tests/   or   python3 tests/test_llm.py
"""

import json
import os
import sys
import tempfile
import time
import types as pytypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.llm import (
    CachedLLM, GeminiLLM, LLMUnavailable, Pacer, RetryingLLM, build_llm, build_llms, is_temporary,
    stopped_message,
    to_gemini_contents,
)

MESSAGES = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first reply"},
    {"role": "user", "content": "that failed a check"},
]


class FakeSDK:
    """Stands in for `google.genai` while a test runs, then restores what was there."""

    def __init__(self, reply="{}", error_code=None, retry_options=False):
        self.calls = []
        self.client_options = None
        sdk = self

        class APIError(Exception):
            def __init__(self, code):
                super().__init__(f"error {code}")
                self.code = code

        class Models:
            def generate_content(self, model, contents, config):
                sdk.calls.append({"model": model, "contents": contents, "config": config})
                if error_code:
                    raise APIError(error_code)
                return pytypes.SimpleNamespace(text=reply)

        class Client:
            def __init__(self, api_key, **options):
                sdk.client_options = options
                self.models = Models()

        record = lambda **fields: pytypes.SimpleNamespace(**fields)
        self.types = pytypes.ModuleType("google.genai.types")
        self.types.Content, self.types.Part, self.types.GenerateContentConfig = record, record, record
        if retry_options:                            # newer SDKs let a client set its own retries
            self.types.HttpOptions, self.types.HttpRetryOptions = record, record
        self.genai = pytypes.ModuleType("google.genai")
        self.genai.Client, self.genai.types = Client, self.types
        self.google = pytypes.ModuleType("google")
        self.google.genai = self.genai

    def __enter__(self):
        names = ("google", "google.genai", "google.genai.types")
        self.saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(dict(zip(names, (self.google, self.genai, self.types))))
        self.saved_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key"
        return self

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if self.saved_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.saved_key


def test_messages_become_gemini_turns_with_the_model_role():
    roles = [turn["role"] for turn in to_gemini_contents(MESSAGES)]
    assert roles == ["user", "model", "user"]
    assert to_gemini_contents(MESSAGES)[1]["parts"] == [{"text": "first reply"}]


def test_gemini_calls_the_sdk_the_documented_way():
    with FakeSDK(reply='{"outcomes": []}') as sdk:
        reply = GeminiLLM(model="gemini-test", requests_per_minute=6000).complete("the rules", MESSAGES)
    call = sdk.calls[0]
    assert reply == '{"outcomes": []}' and call["model"] == "gemini-test"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["config"].system_instruction == "the rules"
    assert call["config"].response_mime_type == "application/json"


def test_a_missing_gemini_model_says_what_to_change():
    with FakeSDK(error_code=404):
        llm = GeminiLLM(model="no-such-model", requests_per_minute=6000)
        try:
            llm.complete("the rules", MESSAGES)
        except RuntimeError as error:
            assert "COURSEALIGN_GEMINI_MODEL" in str(error)
            return
    raise AssertionError("a 404 should explain how to pick another model")


def test_gemini_without_a_key_says_where_to_get_one():
    with FakeSDK():
        del os.environ["GEMINI_API_KEY"]
        saved_google = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            GeminiLLM()
        except SystemExit as stop:
            assert "GEMINI_API_KEY" in str(stop)
            return
        finally:
            if saved_google is not None:
                os.environ["GOOGLE_API_KEY"] = saved_google
    raise AssertionError("a missing key should stop with a clear message")


def test_the_pacer_spaces_requests_out():
    pacer = Pacer(requests_per_minute=1200)           # one request every 0.05 s
    started = time.monotonic()
    for _ in range(3):
        pacer.wait()
    assert time.monotonic() - started >= 0.1           # the 2nd and 3rd each waited a turn


# ── Retries ──────────────────────────────────────────────────────────────────

class Busy(Exception):
    """Looks like an SDK error: the HTTP status is in .code, like Google's."""

    def __init__(self, code=503, text="503 UNAVAILABLE. This model is currently experiencing high demand."):
        super().__init__(text)
        self.code = code


class Flaky:
    """Fails with the given errors in order, then answers."""
    name = "flaky (test)"

    def __init__(self, *errors, reply="{}"):
        self.errors, self.reply, self.calls = list(errors), reply, 0

    def complete(self, system, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.reply


class Clock:
    """Time that only moves when something sleeps."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def retrying(llm, clock, **settings) -> RetryingLLM:
    # rand=1.0 takes the top of each jitter range, so the waits are exact
    return RetryingLLM(llm, sleep=clock.sleep, clock=clock, rand=lambda: 1.0, **settings)


def test_a_busy_model_is_retried_with_waits_that_double():
    clock = Clock()
    llm = retrying(Flaky(Busy(503), Busy(429), reply="ok"), clock, attempts=5)
    assert llm.complete("rules", MESSAGES) == "ok"
    assert clock.sleeps == [2.0, 4.0] and llm.retries == 2


def test_a_model_that_stays_down_raises_llm_unavailable():
    clock = Clock()
    inner = Flaky(*[Busy()] * 4)
    try:
        retrying(inner, clock, attempts=4).complete("rules", MESSAGES)
    except LLMUnavailable as error:
        assert inner.calls == 4 and len(clock.sleeps) == 3
        assert "after 4 tries" in str(error) and "503 UNAVAILABLE" in str(error)
        return
    raise AssertionError("a model that never answers should end in LLMUnavailable")


def test_errors_waiting_cant_fix_are_raised_at_once():
    for code in (400, 401, 403, 404):
        clock, inner = Clock(), Flaky(Busy(code, f"{code} error"))
        try:
            retrying(inner, clock).complete("rules", MESSAGES)
        except LLMUnavailable:
            raise AssertionError(f"a {code} can't be fixed by waiting, so it shouldn't be retried")
        except Busy:
            assert inner.calls == 1 and clock.sleeps == []
            continue
        raise AssertionError(f"a {code} should be raised")


def test_the_wait_the_service_asks_for_is_respected():
    clock = Clock()
    asked = Busy(429, "429 RESOURCE_EXHAUSTED. {'details': [{'retryDelay': '37s'}]}")
    retrying(Flaky(asked, reply="ok"), clock).complete("rules", MESSAGES)
    assert clock.sleeps == [37.0]                     # not the 2 s the backoff alone would wait


def test_after_repeated_failures_calls_pause_then_resume():
    clock = Clock()
    inner = Flaky(*[Busy()] * 4, reply="ok")
    llm = retrying(inner, clock, attempts=2, trip_after=2, cool_down=120.0)
    for _ in range(2):                                # two calls in a row fail every try
        try:
            llm.complete("rules", MESSAGES)
        except LLMUnavailable:
            pass
    try:
        llm.complete("rules", MESSAGES)               # paused: fails at once, without calling the model
    except LLMUnavailable as error:
        assert inner.calls == 4 and "paused" in str(error)
    else:
        raise AssertionError("calls should pause after repeated failures")
    clock.now += 121                                  # the pause is over and the model is back
    assert llm.complete("rules", MESSAGES) == "ok" and llm.gave_up == 3


def google_quota_error(quota_id: str, limit: str = "20") -> Busy:
    """A 429 shaped like the Gemini API's, with the QuotaFailure details it carries."""
    return Busy(429, "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
                     "quota, please check your plan and billing details.', 'status': 'RESOURCE_EXHAUSTED', "
                     "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
                     "[{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', "
                     f"'quotaId': '{quota_id}', 'quotaValue': '{limit}'}}]}}, "
                     "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '41s'}]}}")


def test_a_used_up_daily_limit_stops_at_once_and_says_which_limit():
    clock = Clock()
    inner = Flaky(google_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), reply="ok")
    llm = retrying(inner, clock)
    try:
        llm.complete("rules", MESSAGES)
    except LLMUnavailable as error:
        assert inner.calls == 1 and clock.sleeps == []          # no point waiting for a daily reset
        assert "daily limit" in str(error) and "limit 20" in str(error) and "midnight Pacific" in str(error)
    else:
        raise AssertionError("a used-up daily limit should end in LLMUnavailable")
    try:
        llm.complete("rules", MESSAGES)                         # every later call is skipped
    except LLMUnavailable as error:
        assert inner.calls == 1 and "skipped" in str(error) and "daily limit" in str(error)
    else:
        raise AssertionError("after a daily limit, later calls should be skipped")


def test_a_per_minute_limit_is_still_waited_out():
    clock = Clock()
    inner = Flaky(google_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), reply="ok")
    assert retrying(inner, clock).complete("rules", MESSAGES) == "ok"
    assert clock.sleeps == [41.0]                               # the wait Google asked for


def test_the_reply_cache_saves_paid_answers_and_shows_retry_counts():
    clock = Clock()
    inner = Flaky(Busy(), reply="ok")
    cached = CachedLLM(retrying(inner, clock))
    assert cached.complete("rules", MESSAGES) == "ok" and cached.complete("rules", MESSAGES) == "ok"
    assert inner.calls == 2 and cached.paid_calls == 1         # one busy try, one answer, then the cache
    assert cached.retries == 1                                  # read through the cache


def test_the_stop_message_fits_why_the_run_stopped():
    daily = stopped_message(LLMUnavailable("the daily limit for gemini is used up"), saved=True)
    assert "after the reset" in daily and "only asks for what's missing" in daily
    busy = stopped_message(LLMUnavailable("gemini didn't answer after 7 tries"))
    assert "Wait a few minutes" in busy and "only asks" not in busy


def test_retrying_keeps_the_models_name_so_caches_keep_their_keys():
    assert RetryingLLM(Flaky()).name == "flaky (test)"


def test_lost_connections_are_temporary_and_bad_requests_are_not():
    class APIConnectionError(Exception):              # Anthropic's name for a lost connection
        pass
    assert is_temporary(ConnectionError()) and is_temporary(TimeoutError())
    assert is_temporary(APIConnectionError()) and is_temporary(Busy(529))
    assert not is_temporary(Busy(400)) and not is_temporary(ValueError("bad JSON"))


def test_a_503_from_gemini_is_retried_then_reported_as_unavailable():
    clock = Clock()
    with FakeSDK(error_code=503) as sdk:
        llm = retrying(GeminiLLM(model="gemini-test", requests_per_minute=6000), clock, attempts=3)
        try:
            llm.complete("the rules", MESSAGES)
        except LLMUnavailable:
            assert len(sdk.calls) == 3
            return
    raise AssertionError("a model that keeps returning 503 should end in LLMUnavailable")


def test_gemini_turns_off_the_sdks_own_retries_so_ours_are_the_only_ones():
    with FakeSDK(retry_options=True) as sdk:
        GeminiLLM(model="gemini-test", requests_per_minute=6000)
    assert sdk.client_options["http_options"].retry_options.attempts == 1


def test_every_gemini_request_has_a_time_limit_so_a_hang_cant_freeze_a_run():
    with FakeSDK(retry_options=True) as sdk:
        GeminiLLM(model="gemini-test", requests_per_minute=6000)
    assert sdk.client_options["http_options"].timeout == 120_000          # milliseconds


def test_a_request_that_times_out_is_retried():
    class ReadTimeout(Exception):
        pass
    class TransportError(Exception):                      # httpx's base class for a lost or timed-out request
        pass
    class Hung(ReadTimeout, TransportError):
        pass
    clock = Clock()
    assert retrying(Flaky(Hung("timed out"), reply="ok"), clock).complete("rules", MESSAGES) == "ok"
    assert len(clock.sleeps) == 1


def test_an_sdk_without_retry_options_still_gets_a_client():
    with FakeSDK() as sdk:
        GeminiLLM(model="gemini-test", requests_per_minute=6000)
    assert sdk.client_options == {}


def test_build_llm_wraps_gemini_in_retries():
    with FakeSDK():
        llm = build_llm("gemini")
    assert isinstance(llm, RetryingLLM) and llm.name.startswith("gemini (")


def test_agent1_gets_its_own_model_only_when_one_is_named():
    saved = os.environ.pop("COURSEALIGN_AGENT1_MODEL", None)
    try:
        with FakeSDK():
            reader, judge = build_llms("gemini")
            assert reader is judge                                   # one model unless asked
            os.environ["COURSEALIGN_AGENT1_MODEL"] = "gemini-stronger"
            reader, judge = build_llms("gemini")
        assert reader is not judge and reader.name == "gemini (gemini-stronger)"
        assert isinstance(reader, RetryingLLM) and judge.name != reader.name
    finally:
        os.environ.pop("COURSEALIGN_AGENT1_MODEL", None)
        if saved is not None:
            os.environ["COURSEALIGN_AGENT1_MODEL"] = saved


def test_two_caches_on_one_file_keep_each_others_replies():
    class Named(Flaky):
        def __init__(self, name, reply):
            super().__init__(reply=reply)
            self.name = name
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "replies.json"
        first, second = CachedLLM(Named("model a", "from a"), path), CachedLLM(Named("model b", "from b"), path)
        assert first.complete("rules", MESSAGES) == "from a" and second.complete("rules", MESSAGES) == "from b"
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert len(saved) == 2                                       # neither write erased the other
        again = Named("model a", "never asked")
        assert CachedLLM(again, path).complete("rules", MESSAGES) == "from a" and again.calls == 0


def test_an_unknown_llm_name_is_an_error():
    try:
        build_llm("gpt-0")
    except ValueError as error:
        assert "fake, anthropic, or gemini" in str(error)
        return
    raise AssertionError("an unknown name should be rejected")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
