import pytest

otel = pytest.importorskip("opentelemetry")

from dev_guard.adapters.tracing import OTelTracer


@pytest.mark.integration
def test_otel_tracer_span_smoke():
    tracer = OTelTracer(service_name="dev-guard-test")
    with tracer.span("unit-of-work", component="tests") as span:
        assert span is not None
        assert hasattr(span, "end")

