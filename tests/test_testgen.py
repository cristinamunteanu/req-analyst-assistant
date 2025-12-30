from analysis.testgen import (
    _classify,
    _extract_metrics,
    _performance_templates,
    _ux_templates,
    generate_test_ideas,
)


class TestClassify:
    def test_classify_defaults_to_functional(self):
        assert _classify("No keyword.") == "functional"

    def test_classify_prefers_highest_keyword_count(self):
        text = "The system must prevent hazards and enforce compliance."
        assert _classify(text) == "safety"

    def test_classify_interface_ux(self):
        text = "The UI must support one-handed use with two taps."
        assert _classify(text) == "interface/ux"


class TestExtractMetrics:
    def test_extracts_multiple_metrics(self):
        text = "Respond in <= 300 ms and maintain > 99.5% uptime at 60 FPS."
        assert _extract_metrics(text) == [
            ("<=", "300", "ms"),
            (">", "99.5", "%"),
            ("", "60", "FPS"),
        ]

    def test_extracts_when_no_operator(self):
        text = "Latency is 2 seconds."
        assert _extract_metrics(text) == [("", "2", "s")]

    def test_extracts_empty_when_no_match(self):
        assert _extract_metrics("No metrics here.") == []


class TestPerformanceTemplates:
    def test_uptime_templates_use_percent_threshold(self):
        text = "Service uptime must be >= 99.9% monthly."
        ideas = _performance_templates(text)

        assert ideas[0]["title"] == "Monthly uptime SLO"
        assert "99.9%" in ideas[0]["acceptance"][0]
        assert "99.9%" in ideas[1]["acceptance"][0]

    def test_latency_templates_use_detected_metrics(self):
        text = "p95 latency shall be <= 250 ms."
        ideas = _performance_templates(text)

        assert ideas[0]["title"] == "Latency / response-time budget"
        assert "250 ms" in ideas[0]["acceptance"][0]


class TestUxTemplates:
    def test_taps_budget_is_injected(self):
        ideas = _ux_templates("User completes task in <= 3 taps.")
        assert ideas[0]["acceptance"][0] == "Avg taps per stop ≤ 3"

    def test_taps_budget_default(self):
        ideas = _ux_templates("User completes task quickly.")
        assert ideas[0]["acceptance"][0] == "Avg taps/steps are within the specified budget."


class TestGenerateTestIdeas:
    def test_uses_type_hint_when_valid(self):
        result = generate_test_ideas("Any text", type_hint="safety")
        assert result["type"] == "safety"
        assert len(result["ideas"]) == 3

    def test_invalid_type_hint_falls_back_to_functional(self):
        result = generate_test_ideas("Any text", type_hint="unknown")
        assert result["type"] == "functional"
        assert len(result["ideas"]) == 3

    def test_infers_type_when_no_hint(self):
        result = generate_test_ideas("Uptime should be 99.9% each month.")
        assert result["type"] == "performance"
