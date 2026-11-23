"""
The only place that talks to a language model.

Agents call `complete(system, messages)` and get text back, so swapping the
model, or replacing it with a fake for tests, never touches agent code.

Real models also fail in ordinary ways: a busy server (503, or 529 at
Anthropic), a rate limit (429), a dropped connection. RetryingLLM waits and
tries again; if the model stays down it raises LLMUnavailable, the one model
error the agents handle.
"""

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def parse_json_reply(raw: str) -> object:
    """The JSON object in a model's reply, ignoring any text around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(raw[start:end + 1])


# ── Temporary failures ───────────────────────────────────────────────────────

TEMPORARY_STATUS = {408, 429, 500, 502, 503, 504, 529}     # timed out, rate limited, or busy
NETWORK_ERRORS = {"TransportError", "APIConnectionError"}  # httpx's and Anthropic's lost-connection errors
NO_REPLY = "No reply from the model"                       # how a held pair's reason starts


class LLMUnavailable(RuntimeError):
    """The model kept failing with temporary errors, even after retries.

    Agent 2 holds the pair for a person. Agent 1 lets it stop the run instead:
    a syllabus read with a section missing would make the gap report wrong.
    """


def status_of(error: BaseException) -> int | None:
    """The HTTP status behind an SDK error: Google's SDK calls it .code, Anthropic's .status_code."""
    for name in ("code", "status_code"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def is_temporary(error: BaseException) -> bool:
    """Whether waiting can fix this error. A bad key, a missing model, or a
    malformed request (400, 401, 403, 404) can't be fixed by waiting, so those
    are raised at once rather than retried."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if any(cls.__name__ in NETWORK_ERRORS for cls in type(error).__mro__):
        return True
    return status_of(error) in TEMPORARY_STATUS


def suggested_wait(error: BaseException) -> float | None:
    """How long the service asked us to wait, if it said: a Retry-After
    header, or the retryDelay Google puts in the details of a 429."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        if value:
            return float(value)
    except (AttributeError, TypeError, ValueError):
        pass
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(error))
    return float(match.group(1)) if match else None


def quota_hit(error: BaseException) -> tuple[str, str] | None:
    """The quota a 429 says was used up, from Google's error details:
    (quota id, limit), or None when the error doesn't say."""
    text = str(error)
    found = re.search(r"quotaId['\"]?\s*:\s*['\"]([\w.-]+)", text)
    if not found:
        return None
    limit = re.search(r"quotaValue['\"]?\s*:\s*['\"]?(\d+)", text)
    return found.group(1), (limit.group(1) if limit else "unknown")


def is_daily_quota(error: BaseException) -> bool:
    """A used-up daily quota. Waiting minutes can't fix it: Google resets
    requests-per-day limits at midnight Pacific time."""
    hit = quota_hit(error)
    return status_of(error) == 429 and hit is not None and "PerDay" in hit[0]


def brief(error: BaseException, limit: int = 300) -> str:
    """An error on one line, short enough for a log."""
    text = " ".join(f"{type(error).__name__}: {error}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class AnthropicLLM:
    """Claude through the Anthropic API.

    Needs `pip install anthropic` and an ANTHROPIC_API_KEY environment
    variable. Set COURSEALIGN_MODEL to try another model. Agent 2 calls this
    from several threads at once.

    No temperature here on purpose: Claude Sonnet 5.5 rejects non-default
    sampling settings, so replies can't be forced to repeat. That's why
    every reply is checked by code and why evals should be run more than once.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 16000):
        try:
            import anthropic
        except ImportError as error:
            raise SystemExit("The real LLM needs the Anthropic SDK: pip install anthropic") from error
        self._client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("COURSEALIGN_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens                # room for adaptive thinking plus the reply
        self.name = f"anthropic ({self.model})"

    def complete(self, system: str, messages: list[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        # The reply can include thinking blocks; only text blocks are the answer.
        return "".join(block.text for block in response.content if block.type == "text")


class FakeLLM:
    """Stands in for a model, so tests and offline runs need no API key.

    Two ways to answer:
      replies    canned replies per course id, used in order (a retry gets
                 the next one). Agent 1's fake uses these.
      responder  a function that writes each reply. Agent 2's fake uses one
                 that answers from the hand labels.

    Every call is recorded, so tests can check exactly what an agent saw.
    """

    name = "fake (hand-written replies, not a model)"

    def __init__(self, replies: dict[str, list[str]] | None = None, responder=None, name: str | None = None):
        self._replies = {course: list(queue) for course, queue in (replies or {}).items()}
        self._responder = responder
        self._lock = threading.Lock()               # Agent 2 calls in parallel
        self.calls: list[tuple[str, list[dict]]] = []
        if name:
            self.name = name

    @classmethod
    def from_file(cls, path: str | Path, responder=None, name: str | None = None) -> "FakeLLM":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({
            course: [r if isinstance(r, str) else json.dumps(r) for r in queue]
            for course, queue in raw.items() if not course.startswith("_")
        }, responder=responder, name=name)

    def complete(self, system: str, messages: list[dict]) -> str:
        with self._lock:
            self.calls.append((system, messages))
            first = messages[0]["content"]
            for course, queue in self._replies.items():
                if course in first:
                    if not queue:
                        raise RuntimeError(f"FakeLLM has no replies left for {course}")
                    return queue.pop(0)
        if self._responder is not None:
            return self._responder(system, messages)
        raise RuntimeError("FakeLLM has no reply for this prompt")


_STORES: dict[str, tuple[dict, threading.Lock]] = {}
_STORES_LOCK = threading.Lock()


def _store_for(path: Path | None) -> tuple[dict, threading.Lock]:
    """The saved replies for one cache file, shared by every CachedLLM that uses
    it. Two caches on one file (Agent 1 and Agent 2 on different models) would
    otherwise each write their own copy and erase the other's replies."""
    if path is None:
        return {}, threading.Lock()                 # in memory only, private to one cache
    key = str(path.resolve())
    with _STORES_LOCK:
        if key not in _STORES:
            replies = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            _STORES[key] = (replies, threading.Lock())
        return _STORES[key]


class CachedLLM:
    """Remembers replies by the exact request.

    A re-run then costs nothing for requests that didn't change. Just as
    useful: round 2 differs from round 1 only where a person's decision
    changed something, never because the model answered differently the
    second time. Delete the cache file to ask everything fresh.
    """

    def __init__(self, llm, path: str | Path | None = None):
        self._llm = llm
        self._path = Path(path) if path else None          # None: keep it in memory
        self._replies, self._lock = _store_for(self._path)
        self.name = f"{llm.name}, with a reply cache"
        self.paid_calls = 0                                 # requests the model actually answered

    def __getattr__(self, name: str):
        """Anything else (like RetryingLLM's retry counts) comes from the wrapped model."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._llm, name)

    def complete(self, system: str, messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([self._llm.name, system, messages]).encode()).hexdigest()
        with self._lock:
            if key in self._replies:
                return self._replies[key]
        reply = self._llm.complete(system, messages)
        with self._lock:
            self._replies[key] = reply
            self.paid_calls += 1
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._replies), encoding="utf-8")
        return reply


class Pacer:
    """Spaces requests evenly to stay under a requests-per-minute limit.

    Thread-safe: when Agent 2 judges pairs in parallel, each call takes the
    next free slot and waits for it, instead of all of them firing at once
    and tripping the limit.
    """

    def __init__(self, requests_per_minute: float):
        self._gap = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            start = max(time.monotonic(), self._next)
            self._next = start + self._gap
        time.sleep(max(0.0, start - time.monotonic()))


def to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Our messages in Gemini's shape. Gemini calls the assistant role "model"."""
    return [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages]


class GeminiLLM:
    """Gemini through Google's Gen AI SDK, which has a free tier.

    Needs `pip install google-genai` and a GEMINI_API_KEY from Google AI Studio.
    COURSEALIGN_GEMINI_MODEL picks the model; COURSEALIGN_GEMINI_RPM should match
    your requests-per-minute limit, which AI Studio shows (default: 10).

    Every request is paced to that limit, retries included. The SDK's own
    retries are switched off where the SDK allows it, so RetryingLLM is the
    one retry policy and each retry waits its turn with the pacer.
    """

    def __init__(self, model: str | None = None, requests_per_minute: float | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise SystemExit("Gemini needs Google's Gen AI SDK: pip install google-genai") from error
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY to a key from Google AI Studio.")
        self._client = self._make_client(genai, types, key)
        self._types = types
        self.model = model or os.environ.get("COURSEALIGN_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._pacer = Pacer(requests_per_minute or float(os.environ.get("COURSEALIGN_GEMINI_RPM", "10")))
        self.name = f"gemini ({self.model})"

    @staticmethod
    def _make_client(genai, types, key: str):
        """A client with a time limit on every request and the SDK's own retries off.

        Without a time limit, a request the service never answers waits forever
        and freezes the run. With one, it fails with a timeout, which RetryingLLM
        retries like any other temporary error. COURSEALIGN_REQUEST_TIMEOUT sets
        the limit in seconds (default 120; a reply here usually takes seconds).
        An SDK version without these options gets as many of them as it supports.
        """
        timeout_ms = int(float(os.environ.get("COURSEALIGN_REQUEST_TIMEOUT", "120")) * 1000)
        for options in (
            lambda: types.HttpOptions(timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1)),
            lambda: types.HttpOptions(timeout=timeout_ms),
        ):
            try:
                return genai.Client(api_key=key, http_options=options())
            except (AttributeError, TypeError, ValueError):
                continue
        return genai.Client(api_key=key)

    def complete(self, system: str, messages: list[dict]) -> str:
        types = self._types
        contents = [
            types.Content(role=c["role"], parts=[types.Part(text=part["text"]) for part in c["parts"]])
            for c in to_gemini_contents(messages)
        ]
        self._pacer.wait()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json"),
            )
        except Exception as error:
            if getattr(error, "code", None) == 404:
                raise RuntimeError(
                    f"Gemini has no model called {self.model}. Set COURSEALIGN_GEMINI_MODEL to a model "
                    "your free tier lists in Google AI Studio."
                ) from error
            raise
        # An empty or blocked reply comes back as "", which fails the agents' checks
        # like any other bad reply: they retry, then hand the item to a person.
        return response.text or ""


class RetryingLLM:
    """Retries temporary failures with exponential backoff, then gives up cleanly.

    A busy model, a rate limit, or a dropped connection gets up to `attempts`
    tries. The wait doubles after each failure (1-2 s, 2-4 s, 4-8 s, ... up to
    max_wait), with jitter so parallel calls don't retry in lockstep, and is
    never shorter than what the service asks for. If every try fails, it
    raises LLMUnavailable.

    After `trip_after` calls in a row fail like that, the model is clearly
    down, so calls fail at once for `cool_down` seconds instead of each one
    waiting through its own retries. With Agent 2's parallel calls on a paced
    free tier, that waiting would otherwise add up to many idle minutes.

    A used-up daily quota is different: no wait in this run can fix it, so
    the call gives up at once and every later call is skipped, with a
    message that says which limit it was.

    Any other error is raised at once: retrying can't fix a bad key.
    """

    def __init__(self, llm, attempts: int = 7, first_wait: float = 2.0, max_wait: float = 60.0,
                 trip_after: int = 3, cool_down: float = 120.0,
                 sleep=time.sleep, clock=time.monotonic, rand=random.random):
        self._llm = llm
        self.name = llm.name                    # the same name, so reply caches keep their keys
        self.model = getattr(llm, "model", None)
        self.attempts = max(1, attempts)
        self.first_wait, self.max_wait = first_wait, max_wait
        self.trip_after, self.cool_down = trip_after, cool_down
        self._sleep, self._clock, self._rand = sleep, clock, rand
        self._lock = threading.Lock()           # Agent 2 calls in parallel
        self._failed_in_a_row = 0
        self._paused_until = float("-inf")
        self._last_failure = ""
        self.retries = 0                        # extra tries spent on temporary errors
        self.gave_up = 0                        # calls that ended in LLMUnavailable

    def wait_after(self, attempt: int, error: BaseException) -> float:
        """Seconds to wait after try number `attempt` failed with `error`."""
        backoff = min(self.max_wait, self.first_wait * 2 ** (attempt - 1))
        backoff = backoff / 2 + self._rand() * backoff / 2
        return min(self.max_wait, max(backoff, suggested_wait(error) or 0.0))

    def _check_not_paused(self) -> None:
        with self._lock:
            left = self._paused_until - self._clock()
            if left <= 0:
                return
            self.gave_up += 1
            last = self._last_failure
        if left == float("inf"):
            raise LLMUnavailable(f"{self.name} skipped: {last}")
        raise LLMUnavailable(f"{self.name} skipped: the last {self.trip_after} requests all failed, so calls "
                             f"are paused for {left:.0f} s more. The last failure: {last}")

    def complete(self, system: str, messages: list[dict]) -> str:
        started, error = self._clock(), None
        for attempt in range(1, self.attempts + 1):
            self._check_not_paused()
            try:
                reply = self._llm.complete(system, messages)
            except Exception as caught:
                if not is_temporary(caught):
                    raise
                error = caught
                if is_daily_quota(caught):
                    break                       # retrying for minutes can't fix a daily limit
                if attempt < self.attempts:
                    with self._lock:
                        self.retries += 1
                    self._sleep(self.wait_after(attempt, caught))
                continue
            with self._lock:
                self._failed_in_a_row = 0
            return reply

        hit, daily = quota_hit(error), is_daily_quota(error)
        if daily:
            message = (f"the daily limit for {self.name} is used up ({hit[0]}, limit {hit[1]} requests). "
                       "Google resets it at midnight Pacific time.")
        else:
            message = (f"{self.name} didn't answer after {self.attempts} tries over "
                       f"{self._clock() - started:.0f} s. Last error: {brief(error)}")
            if hit:
                message += f" (Quota: {hit[0]}, limit {hit[1]}.)"
        with self._lock:
            self.gave_up += 1
            self._failed_in_a_row += 1
            self._last_failure = message
            if daily:
                self._paused_until = float("inf")       # nothing more to ask this run
            elif self._failed_in_a_row >= self.trip_after:
                self._paused_until = self._clock() + self.cool_down
        raise LLMUnavailable(message) from error


def _attempts(default: int) -> int:
    """Tries per request: COURSEALIGN_LLM_ATTEMPTS if it's set, else the provider's default."""
    return int(os.environ.get("COURSEALIGN_LLM_ATTEMPTS", default))


def build_llm(name: str):
    """The model behind --llm NAME (or COURSEALIGN_LLM=NAME): anthropic or gemini.

    Both come wrapped in RetryingLLM. Anthropic's SDK already retries each
    request twice with backoff, so Claude gets fewer rounds on top.
    """
    if name == "anthropic":
        return RetryingLLM(AnthropicLLM(), attempts=_attempts(3))
    if name == "gemini":
        return RetryingLLM(GeminiLLM(), attempts=_attempts(7))
    raise ValueError(f"Unknown LLM '{name}'. Use fake, anthropic, or gemini.")


def build_llms(name: str):
    """(Agent 1's model, Agent 2's model): one model unless COURSEALIGN_AGENT1_MODEL
    names another for Agent 1.

    Agent 1 makes only a few calls a run, but an outcome it misses makes every
    later stage wrong (its assignments look like orphans), so it can be worth a
    stronger model with a smaller daily limit. COURSEALIGN_AGENT1_RPM paces it
    (default 5, the free tier's limit for the Flash models).
    """
    judge = build_llm(name)
    reader_model = os.environ.get("COURSEALIGN_AGENT1_MODEL")
    if not reader_model or reader_model == judge.model:
        return judge, judge
    if name == "gemini":
        rpm = float(os.environ.get("COURSEALIGN_AGENT1_RPM", "5"))
        return RetryingLLM(GeminiLLM(model=reader_model, requests_per_minute=rpm), attempts=_attempts(7)), judge
    return RetryingLLM(AnthropicLLM(model=reader_model), attempts=_attempts(3)), judge


def stopped_message(error: LLMUnavailable, saved: bool = False) -> str:
    """What a step prints when the model stayed down and the run had to stop.
    `saved`: the step keeps every reply, so a re-run only asks what's missing."""
    if "daily limit" in str(error):
        advice = ("  Run it again after the reset. Until then, the only way on is another model\n"
                  "  (COURSEALIGN_GEMINI_MODEL) with its own daily limit.\n")
    else:
        advice = ("  This is the model service being busy or a rate limit, not your code or your key.\n"
                  "  Wait a few minutes and run it again. If it keeps happening, check your rate\n"
                  "  limits, or try another model (COURSEALIGN_GEMINI_MODEL or COURSEALIGN_MODEL).\n")
    kept = ("  Every answer this run got is saved, so the next run only asks for what's missing.\n"
            if saved else "")
    return (
        "\n  STOPPED: Agent 1 got no reply from the model. Without that part of the\n"
        "  syllabus the gap report would be wrong, so the run stops instead of guessing.\n\n"
        f"  {error}\n\n" + advice + kept
    )
