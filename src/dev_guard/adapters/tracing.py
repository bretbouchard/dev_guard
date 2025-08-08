from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager


class Tracer(ABC):
    @abstractmethod
    @contextmanager
    def span(self, name: str, **attrs):  # type: ignore[override]
        yield


class OTelTracer(Tracer):
    """OpenTelemetry tracer adapter; requires the tracing extras."""

    def __init__(self, service_name: str = "dev-guard") -> None:
        try:
            from opentelemetry import trace  # type: ignore
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )  # type: ignore
        except Exception as exc:  # pragma: no cover - optional import
            raise RuntimeError(
                "tracing extras not installed. Install with: pip install dev-guard[tracing]"
            ) from exc

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(self, name: str, **attrs):  # type: ignore[override]
        span = self._tracer.start_span(name)
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
        try:
            yield span
        finally:
            span.end()

