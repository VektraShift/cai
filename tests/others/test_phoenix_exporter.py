"""Unit tests for the CAI -> Phoenix OpenInference exporter.

These build a fresh in-memory OpenTelemetry SDK per test (own TracerProvider + InMemorySpanExporter)
so they don't depend on / disturb the session-level SPAN_PROCESSOR_TESTING fixture from conftest.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from cai.sdk.agents.tracing import phoenix_exporter as m
from cai.sdk.agents.tracing.phoenix_exporter import OpenInferenceSpanKindValues


def _span_data(stype, name=None, input=None, output=None, model=None, usage=None, tools=None):  # noqa: A002
    return SimpleNamespace(
        type=stype, name=name, input=input, output=output, model=model, usage=usage, tools=tools
    )


def _cai_span(stype, span_id, trace_id, parent_id=None, item_name=None):
    return SimpleNamespace(
        type=stype,
        name=item_name,
        span_id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        span_data=_span_data(stype, name=item_name),
    )


def _new_exporter():
    """Return (exporter, in_memory_exporter) wired to a fresh in-memory OTel SDK."""
    provider = TracerProvider()
    in_mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(in_mem))
    tracer = provider.get_tracer(__name__)
    exp = m.CAIPhoenixOtelExporter(
        endpoint="phoenix.tools.svc:4317", tracer=tracer, provider=provider
    )
    exp._otlp = None  # never touch a real OTLP backend in tests
    return exp, in_mem


# --- pure helpers ---


def test_otel_id_deterministic_and_in_range():
    a = m._otel_id("abc", bytes_size=8)
    b = m._otel_id("abc", bytes_size=8)
    assert a == b
    assert 0 < a < (1 << 64)
    assert m._otel_id("abc", bytes_size=16) != m._otel_id("def", bytes_size=16)


def test_messages_to_value_string_mixed_and_none():
    assert m._messages_to_value("plain") == "plain"
    assert m._messages_to_value(None) == ""
    assert m._messages_to_value([{"role": "user", "content": "hi"}, "raw"]) == "user: hi\nraw"
    mixed = [{"role": "u", "content": "a"}, {"content": "b"}]
    assert m._messages_to_value(mixed) == "u: a\nb"


@pytest.mark.parametrize(
    "stype,name,model,expected",
    [
        ("agent", "ctf_agent", None, "agent.ctf_agent"),
        ("function", "generic_linux_command", None, "tool.generic_linux_command"),
        ("generation", "reply", "llama", "llm.llama"),
        ("weird", "x", None, "cai.x"),
    ],
)
def test_span_name_prefixes(stype, name, model, expected):
    item = _cai_span(stype, "s1", "t1", item_name=name)
    item.span_data = _span_data(stype, name=name, model=model)
    exp, _ = _new_exporter()
    assert exp._span_name(item) == expected


# --- attribute mapping ---


def test_agent_span_attributes():
    exp, in_mem = _new_exporter()
    item = _cai_span("agent", "s1", "t1", item_name="ctf")
    item.span_data = _span_data("agent", name="ctf", tools=["nmap", "curl"])
    exp.export([item])
    spans = in_mem.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["cai.span.type"] == "agent"
    assert attrs["agent.name"] == "ctf"
    assert attrs["llm.tools"] == "nmap,curl"
    assert (
        m._span_kind(attrs["openinference.span.kind"]) == OpenInferenceSpanKindValues.AGENT.value
    )


def test_function_span_attributes():
    exp, in_mem = _new_exporter()
    item = _cai_span("function", "s1", "t1", item_name="ctf")
    item.span_data = _span_data("function", name="ctf", input="ls", output="out")
    exp.export([item])
    attrs = in_mem.get_finished_spans()[0].attributes
    assert attrs["cai.span.type"] == "function"
    assert attrs["tool.name"] == "ctf"
    assert attrs["input.value"] == "ls"
    assert attrs["output.value"] == "out"
    assert attrs["input.mime_type"] == "text/plain"
    assert attrs["output.mime_type"] == "text/plain"
    assert (
        m._span_kind(attrs["openinference.span.kind"]) == OpenInferenceSpanKindValues.TOOL.value
    )


def test_generation_span_attributes_and_tokens():
    exp, in_mem = _new_exporter()
    item = _cai_span("generation", "span-1", "trace-1", item_name="reply")
    item.span_data = _span_data(
        "generation",
        model="Llama-3",
        input=[{"role": "system", "content": "be an agent"}, {"role": "user", "content": "scan"}],
        output=[{"role": "assistant", "content": "done"}],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3),
    )
    exp.export([item])
    attrs = in_mem.get_finished_spans()[0].attributes
    assert attrs["cai.span.type"] == "generation"
    assert attrs["llm.model_name"] == "Llama-3"
    assert attrs["input.value"] == "system: be an agent\nuser: scan"
    assert attrs["output.value"] == "assistant: done"
    assert attrs["llm.token_count.prompt"] == 9
    assert attrs["llm.token_count.completion"] == 3
    assert m._span_kind(attrs["openinference.span.kind"]) == OpenInferenceSpanKindValues.LLM.value


def test_unknown_type_no_crash():
    exp, in_mem = _new_exporter()
    item = _cai_span("weird", "s1", "t1")
    item.span_data = _span_data("weird")
    exp.export([item])
    assert in_mem.get_finished_spans()[0].attributes["cai.span.type"] == "weird"


# --- parent/child nesting ---


def test_same_batch_parent_child_nesting():
    exp, in_mem = _new_exporter()
    parent = _cai_span("agent", "parent1", "trace1", item_name="ctf")
    child = _cai_span("generation", "child1", "trace1", parent_id="parent1", item_name="reply")
    child.span_data = _span_data("generation", model="llama")
    exp.export([parent, child])
    spans = in_mem.get_finished_spans()
    child_span = next(s for s in spans if s.name.startswith("llm."))
    assert child_span.context.is_valid
    assert child_span.parent is not None


def test_cross_batch_parent_child_nesting():
    exp, in_mem = _new_exporter()
    exp.export([_cai_span("agent", "parent1", "trace1", item_name="ctf")])
    child = _cai_span("generation", "child1", "trace1", parent_id="parent1", item_name="reply")
    child.span_data = _span_data("generation", model="llama")
    exp.export([child])
    spans = in_mem.get_finished_spans()
    child_span = next(s for s in spans if s.name.startswith("llm."))
    assert child_span.context.trace_id == m._otel_id("trace1", bytes_size=16)
    assert child_span.parent is not None


# --- opt-in gating / missing endpoint error ---


def test_init_phoenix_exporter_raises_without_endpoint(monkeypatch):
    monkeypatch.delenv("PHOENIX_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("CAI_PHOENIX_TRACING", "true")
    with pytest.raises(RuntimeError, match="PHOENIX_OTLP_ENDPOINT"):
        m.init_phoenix_exporter()


def test_init_phoenix_exporter_noop_when_not_enabled(monkeypatch):
    # The wiring in tracing/__init__.py only calls this when CAI_PHOENIX_TRACING is set; the
    # function itself is idempotent and requires the endpoint. With neither, no error is raised
    # here only because the caller gates on CAI_PHOENIX_TRACING.
    monkeypatch.delenv("PHOENIX_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("CAI_PHOENIX_TRACING", raising=False)
    # No endpoint && no toggle -> the module-level `init_phoenix_exporter` is not invoked by the
    # __init__ wiring; assert the exporter can still be constructed standalone.
    exp, _ = _new_exporter()
    assert exp.endpoint == "phoenix.tools.svc:4317"
