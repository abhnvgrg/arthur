from __future__ import annotations

import pytest

from arthur.llm import (
    Completion,
    Fragments,
    LLMError,
    RetryPolicy,
    ScriptedLLM,
    ToolCall,
    is_transient,
    parse_arguments,
    retry_after_of,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"city": "Delhi"}', {"city": "Delhi"}),
        ("{}", {}),
        ("", {}),
        (None, {}),
        ({"already": "parsed"}, {"already": "parsed"}),
        ('```json\n{"city": "Delhi"}\n```', {"city": "Delhi"}),
        ('```\n{"city": "Delhi"}\n```', {"city": "Delhi"}),
        ('{"city": "Delhi",}', {"city": "Delhi"}),
    ],
)
async def test_arguments_are_parsed_and_repaired(raw, expected):
    parsed, malformed = parse_arguments(raw)
    assert parsed == expected
    assert malformed is None


@pytest.mark.parametrize("raw", ["not json at all", "{unclosed", "{'single': 'quotes'}"])
async def test_unparseable_arguments_are_reported(raw):
    parsed, malformed = parse_arguments(raw)
    assert parsed == {}
    assert malformed == "Arguments were not valid JSON"


@pytest.mark.parametrize("raw", ["[1, 2, 3]", '"a string"', "42"])
async def test_non_object_arguments_are_reported(raw):
    parsed, malformed = parse_arguments(raw)
    assert parsed == {}
    assert "must be a JSON object" in malformed


async def test_a_completion_without_tool_calls_does_not_want_tools():
    assert Completion(text="hello").wants_tools is False


async def test_a_completion_with_tool_calls_wants_tools():
    completion = Completion(tool_calls=(ToolCall(id="a", name="t", arguments={}),))
    assert completion.wants_tools is True


async def test_the_scripted_llm_returns_completions_in_order():
    first, second = Completion(text="one"), Completion(text="two")
    llm = ScriptedLLM([first, second])

    assert await llm.complete([], []) is first
    assert await llm.complete([], []) is second
    assert llm.exhausted


async def test_the_scripted_llm_records_what_it_was_asked():
    llm = ScriptedLLM([Completion(text="ok")])

    await llm.complete([{"role": "user", "content": "hi"}], [{"name": "t"}])

    assert llm.calls[0]["messages"][0]["content"] == "hi"
    assert llm.calls[0]["tools"] == [{"name": "t"}]


async def test_running_past_the_script_is_an_error():
    llm = ScriptedLLM([])
    with pytest.raises(LLMError):
        await llm.complete([], [])


async def test_a_missing_openai_package_is_a_clean_error(monkeypatch):
    import builtins

    from arthur.llm import OpenAILLM

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(LLMError, match="openai package is not installed"):
        OpenAILLM(api_key="sk-test")._get_client()


async def test_a_missing_api_key_is_a_clean_error(monkeypatch):
    from arthur.llm import OpenAILLM

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAILLM()._get_client()


class Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class ApiError(Exception):
    def __init__(self, status_code, headers=None):
        super().__init__(f"status {status_code}")
        self.response = Response(status_code, headers)


class APITimeoutError(Exception):
    pass


def recorder():
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    return slept, sleep


def policy(attempts=3, **kwargs):
    slept, sleep = recorder()
    return RetryPolicy(attempts=attempts, jitter=False, sleep=sleep, **kwargs), slept


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
async def test_a_transient_status_is_worth_retrying(status):
    assert is_transient(ApiError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_a_client_error_is_not_retried(status):
    assert is_transient(ApiError(status)) is False


async def test_transport_failures_are_transient_without_a_status():
    assert is_transient(APITimeoutError()) is True
    assert is_transient(TimeoutError()) is True
    assert is_transient(ConnectionError()) is True
    assert is_transient(ValueError("bad argument")) is False


async def test_a_retry_after_header_is_read_and_a_missing_one_is_not():
    assert retry_after_of(ApiError(429, {"retry-after": "12"})) == 12.0
    assert retry_after_of(ApiError(429, {"retry-after": "soon"})) is None
    assert retry_after_of(ApiError(429)) is None
    assert retry_after_of(ValueError()) is None


async def test_a_transient_failure_is_retried_until_it_succeeds():
    attempts = []
    retry, slept = policy()

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ApiError(503)
        return "answered"

    assert await retry.run(flaky) == "answered"
    assert len(attempts) == 3
    assert slept == [0.5, 1.0]


async def test_a_bad_request_is_raised_on_the_first_attempt():
    attempts = []
    retry, slept = policy()

    async def rejected():
        attempts.append(1)
        raise ApiError(400)

    with pytest.raises(ApiError):
        await retry.run(rejected)

    assert len(attempts) == 1
    assert slept == []


async def test_the_last_transient_failure_is_raised_not_swallowed():
    attempts = []
    retry, slept = policy(attempts=3)

    async def always_failing():
        attempts.append(1)
        raise ApiError(503)

    with pytest.raises(ApiError):
        await retry.run(always_failing)

    assert len(attempts) == 3
    assert slept == [0.5, 1.0]


async def test_a_retry_after_header_overrides_the_computed_delay():
    retry, slept = policy()
    attempts = []

    async def throttled():
        attempts.append(1)
        if len(attempts) < 2:
            raise ApiError(429, {"retry-after": "3"})
        return "answered"

    await retry.run(throttled)

    assert slept == [3.0]


async def test_a_delay_never_exceeds_the_cap():
    retry, _ = policy(cap=2.0)

    assert retry.delay_for(9) == 2.0
    assert retry.delay_for(0, retry_after=600.0) == 2.0


async def test_jitter_keeps_the_delay_inside_the_window():
    retry = RetryPolicy(base=1.0, cap=8.0, jitter=True)

    for attempt in range(4):
        window = min(8.0, 1.0 * (2**attempt))
        assert 0.0 <= retry.delay_for(attempt) <= window


async def test_a_policy_must_allow_an_attempt():
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)


async def test_one_attempt_means_no_retry():
    attempts = []
    retry, slept = policy(attempts=1)

    async def failing():
        attempts.append(1)
        raise ApiError(503)

    with pytest.raises(ApiError):
        await retry.run(failing)

    assert len(attempts) == 1
    assert slept == []


async def test_the_model_client_retries_and_reports_a_failure_as_an_llm_error():
    from arthur.llm import OpenAILLM

    retry, slept = policy(attempts=2)
    client = OpenAILLM(api_key="sk-test", policy=retry)
    attempts = []

    class Completions:
        async def create(self, **kwargs):
            attempts.append(kwargs)
            raise ApiError(503)

    class Chat:
        completions = Completions()

    class Fake:
        chat = Chat()

    client._client = Fake()

    with pytest.raises(LLMError, match="The model call failed"):
        await client.complete([{"role": "user", "content": "hi"}], [])

    assert len(attempts) == 2
    assert slept == [0.5]


async def test_the_model_client_returns_the_answer_after_one_retry():
    from arthur.llm import OpenAILLM

    retry, _ = policy(attempts=3)
    client = OpenAILLM(api_key="sk-test", policy=retry)
    attempts = []

    class Message:
        content = "hello"
        tool_calls = None

    class Choice:
        message = Message()

    class Answer:
        choices = [Choice()]

    class Completions:
        async def create(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) < 2:
                raise ApiError(429)
            return Answer()

    class Chat:
        completions = Completions()

    class Fake:
        chat = Chat()

    client._client = Fake()

    completion = await client.complete([{"role": "user", "content": "hi"}], [])

    assert completion.text == "hello"
    assert len(attempts) == 2


class Function:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class Fragment:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = Function(name, arguments)


class Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class Chunk:
    def __init__(self, delta=None, choices=None):
        self.choices = choices if choices is not None else [type("C", (), {"delta": delta})()]


class Stream:
    def __init__(self, chunks, fail_after=None):
        self.chunks = list(chunks)
        self.fail_after = fail_after

    async def __aiter__(self):
        for position, chunk in enumerate(self.chunks):
            if self.fail_after is not None and position == self.fail_after:
                raise ApiError(503)
            yield chunk


def streaming_client(chunks, fail_after=None):
    from arthur.llm import OpenAILLM

    sent = []

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return Stream(chunks, fail_after)

    class Chat:
        completions = Completions()

    class Fake:
        chat = Chat()

    client = OpenAILLM(api_key="sk-test", policy=RetryPolicy(attempts=1, sleep=None))
    client._client = Fake()
    return client, sent


async def collect(client, tools=()):
    deltas = []

    async def on_delta(text):
        deltas.append(text)

    completion = await client.stream([{"role": "user", "content": "hi"}], list(tools), on_delta)
    return completion, deltas


async def test_streamed_text_arrives_as_deltas_and_as_one_answer():
    client, sent = streaming_client(
        [Chunk(Delta(content="Hel")), Chunk(Delta(content="lo ")), Chunk(Delta(content="there"))]
    )

    completion, deltas = await collect(client)

    assert deltas == ["Hel", "lo ", "there"]
    assert completion.text == "Hello there"
    assert completion.tool_calls == ()
    assert sent[0]["stream"] is True


async def test_a_stream_with_no_text_has_no_answer():
    client, _ = streaming_client([Chunk(Delta())])

    completion, deltas = await collect(client)

    assert completion.text is None
    assert deltas == []


async def test_a_tool_call_split_across_chunks_is_reassembled():
    client, _ = streaming_client(
        [
            Chunk(Delta(tool_calls=[Fragment(0, id="call_1", name="remem")])),
            Chunk(Delta(tool_calls=[Fragment(0, name="ber")])),
            Chunk(Delta(tool_calls=[Fragment(0, arguments='{"key": "a"')])),
            Chunk(Delta(tool_calls=[Fragment(0, arguments=', "value": "b"}')])),
        ]
    )

    completion, _ = await collect(client)

    assert len(completion.tool_calls) == 1
    call = completion.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "remember"
    assert call.arguments == {"key": "a", "value": "b"}
    assert call.malformed is None


async def test_parallel_streamed_calls_are_kept_apart_and_ordered_by_index():
    client, _ = streaming_client(
        [
            Chunk(Delta(tool_calls=[Fragment(1, id="b", name="convert_units")])),
            Chunk(Delta(tool_calls=[Fragment(0, id="a", name="current_time")])),
            Chunk(Delta(tool_calls=[Fragment(0, arguments="{}"), Fragment(1, arguments="{}")])),
        ]
    )

    completion, _ = await collect(client)

    assert [call.name for call in completion.tool_calls] == ["current_time", "convert_units"]
    assert [call.id for call in completion.tool_calls] == ["a", "b"]


async def test_streamed_arguments_that_never_become_json_are_reported():
    client, _ = streaming_client(
        [Chunk(Delta(tool_calls=[Fragment(0, id="c", name="calculate", arguments="{oops")]))]
    )

    completion, _ = await collect(client)

    assert completion.tool_calls[0].malformed
    assert completion.tool_calls[0].arguments == {}


async def test_a_chunk_with_nothing_in_it_is_skipped():
    client, _ = streaming_client(
        [
            Chunk(choices=[]),
            Chunk(delta=None),
            Chunk(Delta(content="fine")),
        ]
    )

    completion, deltas = await collect(client)

    assert deltas == ["fine"]
    assert completion.text == "fine"


async def test_a_failure_partway_through_a_stream_is_an_llm_error():
    client, _ = streaming_client(
        [Chunk(Delta(content="star")), Chunk(Delta(content="ted"))], fail_after=1
    )

    deltas = []

    async def on_delta(text):
        deltas.append(text)

    with pytest.raises(LLMError, match="The model stream failed"):
        await client.stream([{"role": "user", "content": "hi"}], [], on_delta)

    assert deltas == ["star"]


async def test_the_number_of_streamed_tool_calls_is_capped():
    from arthur.llm import MAX_TOOL_CALLS_PER_STEP

    chunks = [
        Chunk(Delta(tool_calls=[Fragment(n, id=f"c{n}", name="current_time", arguments="{}")]))
        for n in range(MAX_TOOL_CALLS_PER_STEP + 4)
    ]
    client, _ = streaming_client(chunks)

    completion, _ = await collect(client)

    assert len(completion.tool_calls) == MAX_TOOL_CALLS_PER_STEP


async def test_a_fragment_carrying_only_an_id_leaves_the_rest_alone():
    fragments = Fragments()

    fragments.absorb(type("Bare", (), {"id": "only-an-id"})())

    assert fragments.id == "only-an-id"
    assert fragments.name == ""
    assert fragments.arguments == ""
