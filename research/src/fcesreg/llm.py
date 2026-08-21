"""Language model client: disk cache, global ledger, quota governance (§6.11, task C5).

The provider is Groq's OpenAI-compatible endpoint and the model is ``openai/gpt-oss-120b``
on the free tier (amendment G1). Three consequences follow from that, and they shape
everything below.

**There is no Batch API on the free tier, so every call is synchronous.** The 50% batch
discount is gone, and so is the reason §6.11 originally measured latency on a separate
100-call subsample: per-record latency now comes from the very calls that produce the
accuracy figures, which is a stronger measurement than the one it replaces.

**Actual spend is zero, so money is no longer the binding constraint — quota is.**
``CAP_USD`` survives as a runaway guard on *notional* spend, but the limit that will
actually stop a run is the daily token allowance. Cost is therefore reported as measured
token counts costed at a named :class:`RateCard`, recorded on every ledger row together
with its source and the date it was checked, rather than as a bill. That is reproducible
against a public rate card by anyone, and independent of which tier the author was on.

**The disk cache is the checkpoint.** It is written after every call, so a run halted by an
exhausted daily quota loses nothing: re-running the identical set replays from disk, issues
no HTTP request and consumes no quota. There is no second checkpoint mechanism, because a
content-addressed cache already is one.

Two implementation choices worth stating, because both look arbitrary otherwise.

*Raw HTTP rather than the ``openai`` package.* Quota governance is the substance of this
module, and it needs the ``x-ratelimit-*`` headers on successful responses **and** the
``retry-after`` header on a 429. An SDK that translates non-2xx responses into exceptions
hides exactly the values the governance depends on, so the transport speaks the
OpenAI-compatible wire format directly and hands back ``(status, body, headers)``.

*Everything that touches the outside world is injectable.* ``transport``, ``sleep`` and
``monotonic`` are constructor arguments, so the cache, the ledger, the cap, the pacing and
the 429 path are all testable without a network or a real clock.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fcesreg.paths import repo_root, results_path

__all__ = [
    "API_KEY_ENV",
    "CAP_USD",
    "DEFAULT_BASE_URL",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_LIMITS",
    "DEFAULT_MODEL",
    "DEFAULT_RATE_CARD",
    "BudgetExceeded",
    "DailyQuotaExhausted",
    "LLMClient",
    "LLMError",
    "TransportFailure",
    "LLMRequest",
    "LLMResponse",
    "Limits",
    "RateCard",
    "cache_key",
    "read_ledger",
]

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

#: The one environment variable this package reads besides ``FCES_ROOT``. §12.3's Pydantic
#: ``Settings`` rule governs ``system/``; ``fcesreg`` cannot use that class without importing
#: from ``system/``, which the boundary forbids. So: one variable, one accessor, read at call
#: time and never at import time, so merely importing this module cannot fail.
API_KEY_ENV = "GROQ_API_KEY"

#: Runaway protection on notional spend, not a budget. Actual spend on the free tier is $0.00.
CAP_USD = 6.00

#: Anchored against the repository root. A bare relative ``.cache/llm`` would resolve against
#: the working directory, so a run started from ``research/`` would silently miss every cache
#: entry a run from the root had written — the defect ``paths.py`` exists to prevent.
DEFAULT_CACHE_DIR = repo_root() / ".cache" / "llm"

DEFAULT_LEDGER_PATH = results_path("ledger.jsonl")

_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class RateCard:
    """Published prices, carried with every costed figure.

    Actual spend is zero, so a cost figure only means something if the rates that produced
    it travel with it. ``source`` and ``checked`` are written onto every ledger row for that
    reason: the number in the paper is reproducible against a public page on a stated date,
    rather than being a bill that depended on the author's tier.
    """

    model: str
    usd_per_m_input: float
    usd_per_m_output: float
    source: str
    checked: str  # ISO date the source was read

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        """Notional cost of a call at these rates.

        Costed at the **uncached** input rate. The endpoint publishes a lower cached-input
        rate, but whether it applies is decided server-side and is not observable here, so
        assuming it does not errs toward overstating cost — the safe direction for a cost
        claim.
        """
        return (
            input_tokens * self.usd_per_m_input + output_tokens * self.usd_per_m_output
        ) / 1_000_000


#: Verified against Groq's own model page (not a third-party aggregator) on the date shown.
DEFAULT_RATE_CARD = RateCard(
    model=DEFAULT_MODEL,
    usd_per_m_input=0.15,
    usd_per_m_output=0.60,
    source="https://console.groq.com/docs/model/openai/gpt-oss-120b",
    checked="2026-08-08",
)


@dataclass(frozen=True)
class Limits:
    """Free-tier allowances.

    Only two of these four are observable in response headers, and the naming is a trap:
    per Groq's documentation ``x-ratelimit-limit-requests`` is requests per **day** and
    ``x-ratelimit-limit-tokens`` is tokens per **minute**. Requests-per-minute and
    tokens-per-day are therefore *not* exposed by any header and stay configured values,
    verified against the console. The two that headers do report are treated as authoritative
    and overwrite what is configured here the moment a response arrives.
    """

    requests_per_minute: int = 30
    tokens_per_minute: int = 8_000
    requests_per_day: int = 1_000
    tokens_per_day: int = 200_000


DEFAULT_LIMITS = Limits()


class LLMError(RuntimeError):
    """The endpoint returned something this client will not interpret."""


class TransportFailure(LLMError):
    """The request never produced a response: timeout, reset, DNS, connection refused.

    Distinct from an error status because it carries no information about the request's
    validity — it is worth retrying, whereas a 400 is not. Over a sweep lasting days,
    transient network failures are certain rather than possible.
    """


class BudgetExceeded(RuntimeError):
    """Cumulative notional spend crossed ``cap_usd``. Runaway protection, not a bill."""


class DailyQuotaExhausted(RuntimeError):
    """Today's allowance is spent. Stop cleanly; the cache holds everything already done.

    ``completed`` carries the results obtained before the wall was hit, so a caller can
    record a partial run rather than discarding it. Re-running the identical set tomorrow
    replays those from cache for free and resumes on the remainder.
    """

    def __init__(self, message: str, completed: dict[str, LLMResponse] | None = None) -> None:
        super().__init__(message)
        self.completed: dict[str, LLMResponse] = completed or {}


@dataclass(frozen=True)
class LLMRequest:
    """One adjudication or classification. ``custom_id`` is how the caller finds it again."""

    custom_id: str
    system: str
    prompt: str
    max_tokens: int = 64
    json_schema: dict[str, Any] | None = None
    condition: str = "unspecified"


@dataclass(frozen=True)
class LLMResponse:
    """One completion.

    ``input_tokens`` and ``output_tokens`` are the real counts for this request and response
    whether it was served live or from cache. ``cache_hit`` says whether it consumed quota.
    ``usd`` is the notional cost **consumed** and is ``0.0`` on a cache hit; ``usd_uncached``
    is the notional cost of those tokens regardless of cache — what a fresh run would pay.
    The method's cost per thousand records sums ``usd_uncached``, because caching is our
    development convenience and not a property of the method; quota accounting sums tokens
    where ``cache_hit`` is false.
    """

    text: str
    input_tokens: int
    output_tokens: int
    usd: float
    usd_uncached: float
    latency_ms: int
    cache_hit: bool
    model: str
    finish_reason: str


def cache_key(
    model: str, system: str, prompt: str, json_schema: dict[str, Any] | None = None
) -> str:
    """``sha256(model + system + prompt + str(json_schema))``, exactly as §6.11 specifies.

    The model is inside the digest, so changing checkpoints cannot silently reuse completions
    produced by a different model — the same invalidation trap ``embed.cache_key`` guards.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(system.encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    digest.update(str(json_schema).encode("utf-8"))
    return digest.hexdigest()


def read_ledger(path: Path | str = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Every well-formed row. A torn final line is skipped rather than raising.

    Appends are atomic, so a torn line should be impossible; if one appears anyway, losing a
    cost row is preferable to making the ledger unreadable for every subsequent run.
    """
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise LLMError(
            f"{API_KEY_ENV} is unset. Put it in .env (gitignored) or export it; "
            "it is never committed to the repository."
        )
    return key


def _http_transport(
    base_url: str, timeout: float
) -> Callable[[dict[str, Any]], tuple[int, dict[str, Any], Mapping[str, str]]]:
    """Speak the OpenAI-compatible wire format directly, returning status and headers.

    ``httpx`` is imported lazily so that importing this module, replaying a fully cached run,
    or running the test suite never requires the dependency or a network stack.
    """

    def transport(payload: dict[str, Any]) -> tuple[int, dict[str, Any], Mapping[str, str]]:
        import httpx

        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            # Translated at the boundary so `_call` can retry it without importing httpx,
            # which stays a lazy dependency. A read timeout on a long reasoning generation
            # is ordinary, and left unhandled one of them ends a multi-day sweep.
            raise TransportFailure(f"{type(exc).__name__}: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": response.text[:500]}}
        return response.status_code, body, response.headers

    return transport


def _shard(cache_dir: Path, key: str) -> Path:
    # Two levels deep, as in embed.py: a flat directory of several hundred thousand entries
    # is slow to stat on most filesystems.
    return cache_dir / key[:2] / key[2:4] / f"{key}.json"


@dataclass
class _Observed:
    """What the last response's headers reported. ``None`` until a response has arrived."""

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None


class LLMClient:
    """Cache, ledger and quota governance around one synchronous completion endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        ledger_path: Path | str = DEFAULT_LEDGER_PATH,
        run_id: str | None = None,
        cap_usd: float = CAP_USD,
        rate_card: RateCard = DEFAULT_RATE_CARD,
        limits: Limits = DEFAULT_LIMITS,
        base_url: str = DEFAULT_BASE_URL,
        transport: Callable[[dict[str, Any]], tuple[int, dict[str, Any], Mapping[str, str]]]
        | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_retries: int = 5,
        # A reasoning model asked for up to 1024 output tokens can legitimately take longer
        # than a minute; 60s was timing out on the hardest pairs of the cascade sweep.
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.ledger_path = Path(ledger_path)
        self.run_id = run_id
        self.cap_usd = cap_usd
        self.rate_card = rate_card
        self.limits = limits
        self.max_retries = max_retries
        self._transport = transport or _http_transport(base_url, timeout)
        self._sleep = sleep
        self._monotonic = monotonic

        # Build the cache and ledger before the first call, never after (§6.11 rule 4).
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        self.observed = _Observed()
        self._token_window: deque[tuple[float, int]] = deque()
        self._last_request_at: float | None = None

        # Seeded from the shared ledger, which is exactly why the ledger is global: a cap that
        # reset with every run would not be a cap, and today's remaining allowance depends on
        # what earlier runs already consumed.
        history = read_ledger(self.ledger_path)
        self._spend_usd = sum(float(r.get("usd", 0.0)) for r in history)
        self._tokens_today = _tokens_consumed_on(history, datetime.now(UTC).date().isoformat())
        self._requests_today = sum(
            1
            for r in history
            if not r.get("cache_hit", False)
            and str(r.get("ts", "")).startswith(datetime.now(UTC).date().isoformat())
        )

    # -- public ---------------------------------------------------------------------

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 64,
        json_schema: dict[str, Any] | None = None,
        condition: str = "unspecified",
    ) -> LLMResponse:
        """Cache hit returns immediately having consumed nothing; a miss calls and records.

        The ledger row is appended for both outcomes. A hit that wrote no row would make the
        C5 criterion uncheckable — "every row logs ``cache_hit=true``" needs the rows to exist.
        """
        key = cache_key(self.model, system, prompt, json_schema)
        cached = self._read_cache(key)
        if cached is not None:
            response = LLMResponse(
                text=cached["text"],
                input_tokens=int(cached["input_tokens"]),
                output_tokens=int(cached["output_tokens"]),
                usd=0.0,
                usd_uncached=self.rate_card.usd(
                    int(cached["input_tokens"]), int(cached["output_tokens"])
                ),
                latency_ms=0,
                cache_hit=True,
                model=cached.get("model", self.model),
                finish_reason=cached.get("finish_reason", "stop"),
            )
            self._append_ledger(key, response, condition)
            return response

        if self._spend_usd > self.cap_usd:
            raise BudgetExceeded(
                f"notional spend ${self._spend_usd:.4f} already exceeds cap ${self.cap_usd:.2f}"
            )

        body = self._call(system, prompt, max_tokens, json_schema)
        response = LLMResponse(
            text=body["text"],
            input_tokens=body["input_tokens"],
            output_tokens=body["output_tokens"],
            usd=self.rate_card.usd(body["input_tokens"], body["output_tokens"]),
            usd_uncached=self.rate_card.usd(body["input_tokens"], body["output_tokens"]),
            latency_ms=body["latency_ms"],
            cache_hit=False,
            model=self.model,
            finish_reason=body["finish_reason"],
        )

        # Cache before the ledger: a crash between the two costs a duplicated ledger row on the
        # next run, whereas the reverse loses the completion the quota already paid for.
        self._write_cache(key, response)
        self._append_ledger(key, response, condition)

        if self._spend_usd > self.cap_usd:
            raise BudgetExceeded(
                f"notional spend ${self._spend_usd:.4f} crossed cap ${self.cap_usd:.2f}"
            )
        return response

    def complete_many(self, requests: Sequence[LLMRequest]) -> dict[str, LLMResponse]:
        """Synchronous, paced and resumable. Keyed by ``custom_id``, never by position.

        Cached requests are served first, before any quota is spent. Two reasons: a resumed
        run then shows what work actually remains before committing a single token, and a
        quota stop can never strand a result that was already free to obtain. That reordering
        is also why returning a list positionally would be a bug waiting to happen.

        On exhaustion this raises :class:`DailyQuotaExhausted` carrying everything completed
        so far, rather than spending the remaining allowance on requests that cannot finish.
        """
        cached_first: list[LLMRequest] = []
        remainder: list[LLMRequest] = []
        for request in requests:
            key = cache_key(self.model, request.system, request.prompt, request.json_schema)
            (cached_first if _shard(self.cache_dir, key).exists() else remainder).append(request)

        results: dict[str, LLMResponse] = {}
        for request in [*cached_first, *remainder]:
            try:
                results[request.custom_id] = self.complete(
                    request.system,
                    request.prompt,
                    request.max_tokens,
                    request.json_schema,
                    request.condition,
                )
            except DailyQuotaExhausted as exhausted:
                raise DailyQuotaExhausted(
                    f"{exhausted}; {len(results)}/{len(requests)} completed and cached",
                    completed=results,
                ) from exhausted
        return results

    # -- cache ----------------------------------------------------------------------

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = _shard(self.cache_dir, key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # A truncated write from an interrupted run. Treat as a miss.
        if payload.get("schema") != _CACHE_SCHEMA:
            return None
        return payload

    def _write_cache(self, key: str, response: LLMResponse) -> None:
        path = _shard(self.cache_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _CACHE_SCHEMA,
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "model": response.model,
            "finish_reason": response.finish_reason,
        }
        # Write-then-rename, so an interrupted run leaves no half-written entry that a later
        # run would read back as a real completion.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    # -- ledger ---------------------------------------------------------------------

    def _append_ledger(self, key: str, response: LLMResponse, condition: str) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "condition": condition,
            "prompt_sha256": key,
            "provider": "groq",
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "usd": round(response.usd, 8),
            "usd_uncached": round(response.usd_uncached, 8),
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "rate_usd_per_m_input": self.rate_card.usd_per_m_input,
            "rate_usd_per_m_output": self.rate_card.usd_per_m_output,
            "rate_card_checked": self.rate_card.checked,
        }
        # One write() of one line ending in \n. The sweep runs several conditions and a torn
        # line loses a cost row.
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        self._spend_usd += response.usd
        if not response.cache_hit:
            self._tokens_today += response.input_tokens + response.output_tokens
            self._requests_today += 1

    # -- quota ----------------------------------------------------------------------

    def _note_headers(self, headers: Mapping[str, str]) -> None:
        """Headers are the runtime source of truth and overwrite what was configured."""

        def as_int(name: str) -> int | None:
            raw = headers.get(name)
            if raw is None:
                return None
            try:
                return int(float(str(raw).strip()))
            except ValueError:
                return None

        self.observed.remaining_requests = as_int("x-ratelimit-remaining-requests")
        self.observed.remaining_tokens = as_int("x-ratelimit-remaining-tokens")
        self.observed.limit_requests = as_int("x-ratelimit-limit-requests")
        self.observed.limit_tokens = as_int("x-ratelimit-limit-tokens")

        # limit-requests is per DAY and limit-tokens is per MINUTE. Getting this pair the wrong
        # way round would pace at a thirtieth of the real rate, or blow through the daily
        # allowance in a minute.
        if self.observed.limit_tokens is not None:
            self.limits = Limits(
                requests_per_minute=self.limits.requests_per_minute,
                tokens_per_minute=self.observed.limit_tokens,
                requests_per_day=self.observed.limit_requests or self.limits.requests_per_day,
                tokens_per_day=self.limits.tokens_per_day,
            )
        elif self.observed.limit_requests is not None:
            self.limits = Limits(
                requests_per_minute=self.limits.requests_per_minute,
                tokens_per_minute=self.limits.tokens_per_minute,
                requests_per_day=self.observed.limit_requests,
                tokens_per_day=self.limits.tokens_per_day,
            )

    def _estimate_tokens(
        self, system: str, prompt: str, max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> int:
        """A pacing estimate only. The ledger always records measured counts.

        Four characters per token is crude, but it is used solely to decide how long to wait,
        and waiting slightly too long costs time while waiting too little costs a 429.

        **``json_schema`` counts too.** It rides in ``response_format`` on the wire and the
        endpoint bills it as part of the request, but it is not part of ``system`` or
        ``prompt`` -- omitting it here undercounts real usage silently rather than loudly.
        For ``adjudicate``'s two-field schema with no enum this was negligible and the gap
        went unnoticed; a classification schema constraining `code`/`runner_up` to an N-code
        enum (twice, once per field) is not, and the client paced itself as if the request
        were far smaller than the server saw, which the server then corrected with a real
        429 no amount of clean code on this side avoids without counting it here.
        """
        schema_chars = len(json.dumps(json_schema)) if json_schema is not None else 0
        return (len(system) + len(prompt) + schema_chars) // 4 + max_tokens

    def _wait_for_quota(self, estimated_tokens: int) -> None:
        if self.observed.remaining_requests == 0:
            raise DailyQuotaExhausted("no requests remaining today (per response headers)")
        if self._tokens_today + estimated_tokens > self.limits.tokens_per_day:
            raise DailyQuotaExhausted(
                f"daily token allowance would be exceeded: {self._tokens_today} used of "
                f"{self.limits.tokens_per_day}, next call needs about {estimated_tokens}"
            )
        if self._requests_today >= self.limits.requests_per_day:
            raise DailyQuotaExhausted(
                f"daily request allowance spent: {self._requests_today} of "
                f"{self.limits.requests_per_day}"
            )

        if estimated_tokens > self.limits.tokens_per_minute:
            # Waiting cannot help: the request alone exceeds a full minute's allowance. Fail
            # loudly rather than spinning in the pacing loop below.
            raise LLMError(
                f"a single call needs about {estimated_tokens} tokens, above the "
                f"per-minute allowance of {self.limits.tokens_per_minute}"
            )

        now = self._monotonic()

        # Requests per minute: keep a floor under the gap between consecutive calls.
        if self.limits.requests_per_minute > 0 and self._last_request_at is not None:
            gap = 60.0 / self.limits.requests_per_minute
            wait = gap - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._monotonic()

        # Tokens per minute: a rolling sixty-second window, so a burst waits for the oldest
        # call to age out rather than being rejected.
        self._expire_window(now)
        while (
            sum(tokens for _, tokens in self._token_window) + estimated_tokens
            > self.limits.tokens_per_minute
            and self._token_window
        ):
            oldest = self._token_window[0][0]
            self._sleep(max(0.0, 60.0 - (now - oldest)) + 0.01)
            now = self._monotonic()
            self._expire_window(now)

    def _expire_window(self, now: float) -> None:
        while self._token_window and now - self._token_window[0][0] >= 60.0:
            self._token_window.popleft()

    # -- transport ------------------------------------------------------------------

    def _call(
        self, system: str, prompt: str, max_tokens: int, json_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        estimated = self._estimate_tokens(system, prompt, max_tokens, json_schema)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            # Determinism is a standing rule. It is not guaranteed by any endpoint, but an
            # unset temperature would not even ask for it.
            "temperature": 0.0,
        }
        if json_schema is not None:
            # `json_schema` is the whole `response_format.json_schema` object, so it must
            # carry `name` alongside `schema`. Checked here rather than left to the
            # endpoint: a 400 arrives only after the request is issued, and on a sweep that
            # means discovering it once per severity instead of once, before anything runs.
            if "name" not in json_schema or "schema" not in json_schema:
                raise LLMError(
                    "json_schema must be the response_format.json_schema object, with "
                    f"'name' and 'schema' keys; got {sorted(json_schema)}. Passing the "
                    "bare JSON Schema is rejected by the endpoint with a 400."
                )
            payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        for attempt in range(self.max_retries + 1):
            self._wait_for_quota(estimated)
            started = self._monotonic()
            try:
                status, body, headers = self._transport(payload)
            except TransportFailure as exc:
                if attempt == self.max_retries:
                    raise LLMError(
                        f"transport failed {self.max_retries} times in a row: {exc}"
                    ) from exc
                self._sleep(2.0 * (attempt + 1))
                continue
            elapsed_ms = int((self._monotonic() - started) * 1000)
            self._note_headers(headers)
            self._last_request_at = self._monotonic()

            if status == 200:
                usage = body.get("usage") or {}
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                self._token_window.append((self._monotonic(), input_tokens + output_tokens))
                choices = body.get("choices") or []
                if not choices:
                    raise LLMError(f"200 response carried no choices: {body!r}")
                return {
                    "text": (choices[0].get("message") or {}).get("content") or "",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": elapsed_ms,
                    "finish_reason": choices[0].get("finish_reason") or "stop",
                }

            if status == 429:
                if self.observed.remaining_requests == 0:
                    raise DailyQuotaExhausted("429 with no requests remaining today")
                if attempt == self.max_retries:
                    raise LLMError(f"rate limited after {self.max_retries} retries")
                # Honour retry-after rather than retrying blindly; fall back to a widening
                # backoff only when the header is absent.
                self._sleep(_retry_after(headers, default=2.0 * (attempt + 1)))
                continue

            if status >= 500:
                if attempt == self.max_retries:
                    raise LLMError(f"server error {status} after {self.max_retries} retries")
                self._sleep(2.0 * (attempt + 1))
                continue

            # A structured-output generation that failed to validate is the model missing
            # its own schema on this attempt, not a malformed request: the identical payload
            # usually succeeds on a retry. It is a 400, so it would otherwise be fatal — and
            # on a sweep of thousands of calls, one such generation would kill a multi-day
            # run that is otherwise entirely recoverable from cache. Everything else in the
            # 4xx range stays fatal, because retrying a genuinely bad request only wastes
            # quota.
            if status == 400 and _is_json_validate_failure(body):
                if attempt == self.max_retries:
                    raise LLMError(
                        f"the model failed to produce schema-valid JSON {self.max_retries} "
                        f"times in a row for one request; this is the pathological case, "
                        f"not the transient one: {body!r}"
                    )
                self._sleep(1.0 * (attempt + 1))
                continue

            raise LLMError(f"{status} from endpoint: {body!r}")

        raise LLMError("retry loop exited without a response")  # pragma: no cover


def _is_json_validate_failure(body: Any) -> bool:
    """Whether a 400 is the endpoint reporting a bad *generation* rather than a bad request.

    Matched on the error code, not on the human-readable message, which is prose and will
    be reworded without notice.
    """
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    return isinstance(error, dict) and error.get("code") == "json_validate_failed"


def _retry_after(headers: Mapping[str, str], default: float) -> float:
    raw = headers.get("retry-after")
    if raw is None:
        return default
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return default


def _tokens_consumed_on(rows: Iterable[dict[str, Any]], iso_date: str) -> int:
    """Tokens that actually hit the endpoint on ``iso_date``. Cache hits consumed nothing."""
    return sum(
        int(r.get("input_tokens", 0)) + int(r.get("output_tokens", 0))
        for r in rows
        if not r.get("cache_hit", False) and str(r.get("ts", "")).startswith(iso_date)
    )
