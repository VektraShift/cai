# CAIPhoenixOtelExporter
# A `cai.sdk.agents.tracing.TracingExporter` that exports CAI agent/function/generation spans to
# Arize Phoenix via OTLP (HTTP/protobuf) with OpenInference semantic conventions.
#
# Vendored so it is copied into the CAI fork at build time (build-time patch, not a sitecustomize hook).
# The CAI-package imports are done lazily inside `init_phoenix_exporter()` so this module also imports
# standalone (which the unit test in `tests/test_cai_phoenix_exporter.py` relies on to inspect the
# emitted spans via an in-memory OpenTelemetry SDK).
from __future__ import annotations

import hashlib
import os
from typing import Any

import opentelemetry.context as otel_ctx
import opentelemetry.trace as otel_trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState, set_span_in_context

try:  # openinference may not be installed in a bare test/probe environment; unknown types degrade gracefully
    from openinference.semconv.trace import OpenInferenceSpanKindValues
except Exception:  # pragma: no cover
    OpenInferenceSpanKindValues = None

try:  # make the class importable standalone (tests) yet a real TracingExporter inside the fork
    from .processor_interface import TracingExporter
except Exception:  # pragma: no cover
    TracingExporter = object


def _otel_id(value: str, *, bytes_size: int) -> int:
    """Deterministically map a CAI string id to an OTel int id (128-bit trace / 64-bit span)."""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:bytes_size], byteorder="big")


def _make_context(span_ctx: SpanContext) -> otel_ctx.Context:
    """Build an OTel context carrying a parent span context (for cross-batch nesting)."""
    return set_span_in_context(NonRecordingSpan(span_ctx))


def _messages_to_value(data: Any) -> str:
    """Best-effort serialization of CAI generation input/output (list of role/content dicts)."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (list, tuple)):
        parts: list[str] = []
        for item in data:
            if isinstance(item, dict):
                role = item.get("role") or ""
                name = item.get("name") or ""
                content = item.get("content") or ""
                text = content if isinstance(content, str) else str(content)
                if name:
                    parts.append(f"{role}: {name}: {text}")
                elif role:
                    parts.append(f"{role}: {text}")
                else:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(data)


def _span_kind(value: Any) -> str:
    """Return the canonical string for an OpenInference span kind (or fall back to str)."""
    if value is None:
        return "unknown"
    raw = getattr(value, "value", value)
    return str(raw)


class CAIPhoenixOtelExporter(TracingExporter):
    """A `cai.sdk.agents.tracing.TracingExporter` that exports CAI spans to Phoenix via OTLP."""

    def __init__(self, endpoint: str | None = None, tracer: Any = None, provider: Any = None):
        self.endpoint = endpoint or os.environ.get(
            "PHOENIX_OTLP_ENDPOINT", "phoenix.tools.svc:4317"
        )
        # The tracer MUST come from a real SDK TracerProvider. Using the global tracer returns
        # NonRecordingSpan (no .resource), which the OTLP encoder rejects -> nothing reaches Phoenix.
        # Phoenix serves OTLP over gRPC (port 4317); its OTLP HTTP (4318) is not listening.
        if tracer is not None:
            self._tracer = tracer
            self._provider = provider
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            self._provider = TracerProvider()
            # gRPC endpoint is host:port (no scheme); insecure for the in-cluster LAN OTLP.
            self._otlp = OTLPSpanExporter(endpoint=self.endpoint, insecure=True, timeout=5)
            self._provider.add_span_processor(BatchSpanProcessor(self._otlp))
            self._tracer = self._provider.get_tracer("cai-phoenix-otel")

    def export(self, items: list[Any]) -> None:  # type: ignore[override]
        for item in items:
            try:
                self._export_one(item)
            except Exception as exc:  # non-fatal, never stall a CAI run
                print(f"[cai-phoenix-otel] export error (non-fatal): {exc}", file=os.sys.stderr)
        # Spans are picked up by the TracerProvider's processor (OTLP/Batch). Flush so a short
        # API run still ships its spans promptly.
        if getattr(self, "_provider", None) is not None and hasattr(self._provider, "force_flush"):
            try:
                self._provider.force_flush()
            except Exception:  # pragma: no cover
                pass

    def _export_one(self, item: Any) -> Any:
        name = self._span_name(item)
        trace_id = _otel_id(getattr(item, "trace_id", name), bytes_size=16)
        parent_id = getattr(item, "parent_id", None)

        ctx = None
        if parent_id is not None:
            # TraceFlags in some OTel versions resolves to a plain int (loses `.sampled`); wrap it
            # so the SDK can read `trace_flags.sampled` when it links the child to its parent.
            ctx = _make_context(
                SpanContext(
                    trace_id=trace_id,
                    span_id=_otel_id(parent_id, bytes_size=8),
                    is_remote=False,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    trace_state=TraceState(),
                )
            )

        span = self._tracer.start_span(name, context=ctx)
        span.set_attribute("cai.span", True)
        span.set_attribute("parent_id", str(parent_id))
        self._span_from_item(item, span)
        span.end()
        return span

    def _span_name(self, item: Any) -> str:
        sd = getattr(item, "span_data", None)
        stype = getattr(sd, "type", None) or getattr(item, "type", None)
        base = getattr(item, "name", None) or getattr(sd, "name", None) or "span"
        if stype == "agent":
            return f"agent.{base}"
        if stype == "function":
            return f"tool.{base}"
        if stype == "generation":
            model = getattr(sd, "model", None) or base
            return f"llm.{model}"
        return f"cai.{base}"

    def _span_from_item(self, item: Any, span: Any) -> None:
        span_data = getattr(item, "span_data", None)
        if span_data is None:
            return
        self._apply_span(span_data, span)

    def _apply_span(self, data: Any, span: Any) -> None:
        kind = OpenInferenceSpanKindValues
        stype = getattr(data, "type", None) or getattr(data, "kind", None)
        if stype == "agent":
            span.set_attribute("cai.span.type", "agent")
            span.set_attribute("openinference.span.kind", _span_kind(kind.AGENT))
            if getattr(data, "name", None):
                span.set_attribute("agent.name", str(data.name))
            if getattr(data, "tools", None):
                span.set_attribute("llm.tools", data.tools if isinstance(data.tools, str) else ",".join(data.tools))
        elif stype == "function":
            span.set_attribute("cai.span.type", "function")
            span.set_attribute("openinference.span.kind", _span_kind(kind.TOOL))
            tool_name = getattr(data, "name", None)
            if tool_name:
                span.set_attribute("tool.name", str(tool_name))
            self._set_input_output(data, span)
        elif stype == "generation":
            span.set_attribute("cai.span.type", "generation")
            span.set_attribute("openinference.span.kind", _span_kind(kind.LLM))
            model = getattr(data, "model", None)
            if model:
                span.set_attribute("llm.model_name", str(model))
            self._set_input_output(data, span)
            usage = getattr(data, "usage", None) or {}
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is None and isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None and isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
            if prompt_tokens is not None:
                span.set_attribute("llm.token_count.prompt", int(prompt_tokens))
            if completion_tokens is not None:
                span.set_attribute("llm.token_count.completion", int(completion_tokens))
        else:
            span.set_attribute("cai.span.type", str(stype or "unknown"))

    @staticmethod
    def _set_input_output(data: Any, span: Any) -> None:
        input_val = getattr(data, "input", None)
        output_val = getattr(data, "output", None)
        if input_val is not None:
            span.set_attribute("input.value", _messages_to_value(input_val))
            span.set_attribute("input.mime_type", "text/plain")
        if output_val is not None:
            span.set_attribute("output.value", _messages_to_value(output_val))
            span.set_attribute("output.mime_type", "text/plain")


def init_phoenix_exporter() -> None:
    """Register the Phoenix exporter as CAI's tracing processor (replacing the OpenAI backend).

    Requires ``CAI_PHOENIX_TRACING=1`` (the caller checks this) AND ``PHOENIX_OTLP_ENDPOINT`` set.
    If the toggle is on but the endpoint is missing, this raises so the misconfiguration is visible
    instead of silently exporting nothing.
    """
    endpoint = os.environ.get("PHOENIX_OTLP_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "CAI_PHOENIX_TRACING is set but PHOENIX_OTLP_ENDPOINT is not. "
            "Set PHOENIX_OTLP_ENDPOINT to the OTLP gRPC endpoint of your Phoenix collector "
            "(e.g. 'phoenix.tools.svc:4317') to export traces."
        )
    try:
        from .setup import GLOBAL_TRACE_PROVIDER
        from .processors import BatchTraceProcessor

        GLOBAL_TRACE_PROVIDER.set_processors([BatchTraceProcessor(CAIPhoenixOtelExporter(endpoint))])
        print(f"[cai-phoenix-otel] exporter registered -> {endpoint}", file=os.sys.stderr)
    except Exception as exc:
        raise RuntimeError(f"[cai-phoenix-otel] init failed: {exc}") from exc
