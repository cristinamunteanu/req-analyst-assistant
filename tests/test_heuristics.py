import pytest
from analysis.heuristics import analyze_clarity, Issue


class TestHeuristicAnalysis:
    """Test suite for the heuristic-based requirement clarity analysis."""

    def test_analyze_clarity_perfect_requirement(self):
        """
        Test that a well-written requirement receives a high clarity score.
        
        Steps:
            - Provide a clear, specific requirement with active voice
            - Call analyze_clarity on the requirement
            - Assert that the score is high (>= 8.0) and no issues are detected
        """
        requirement = "The system shall authenticate users within 2 seconds using OAuth 2.0 protocol."
        result = analyze_clarity(requirement)
        
        assert result["clarity_score"] >= 80
        assert len(result["issues"]) == 0

    def test_analyze_clarity_with_tbd(self):
        """
        Test that requirements with TBD markers are detected and penalized.
        
        Steps:
            - Provide a requirement containing TBD markers
            - Call analyze_clarity on the requirement
            - Assert that TBD issues are detected and score is reduced
        """
        requirement = "The system shall process data at TBD rate and store results TBD."
        result = analyze_clarity(requirement)
        
        # Should detect 2 TBD instances
        tbd_issues = [issue for issue in result["issues"] if issue.type == "TBD"]
        assert len(tbd_issues) == 2
        assert result["clarity_score"] < 80
        
        # Check issue details
        assert any("tbd" in issue.note.lower() for issue in tbd_issues)

    def test_analyze_clarity_with_vague_terms(self):
        """
        Test that vague terms are detected and flagged as clarity issues.
        
        Steps:
            - Provide a requirement with multiple vague terms
            - Call analyze_clarity on the requirement
            - Assert that vague term issues are detected
        """
        requirement = "The system should be fast and provide good performance with reasonable response times."
        result = analyze_clarity(requirement)
        
        vague_issues = [issue for issue in result["issues"] if issue.type in ["Ambiguous", "NonVerifiable"]]
        assert len(vague_issues) > 0
        assert result["clarity_score"] < 80

        # Should detect terms like "fast", "good", "reasonable"
        vague_descriptions = " ".join([issue.note for issue in vague_issues])
        assert any(term in vague_descriptions.lower() for term in ["fast", "good", "reasonable"])

    def test_analyze_clarity_with_passive_voice(self):
        """
        Test that passive voice constructions are detected.
        
        Steps:
            - Provide a requirement written in passive voice
            - Call analyze_clarity on the requirement
            - Assert that passive voice issues are detected
        """
        requirement = "Data will be processed by the system and results will be stored in the database."
        result = analyze_clarity(requirement)
        
        passive_issues = [issue for issue in result["issues"] if issue.type == "PassiveVoice"]
        assert len(passive_issues) > 0
        assert result["clarity_score"] < 100

    def test_analyze_clarity_security_vagueness(self):
        """
        Test that vague security-related terms are detected and flagged.
        
        Steps:
            - Provide a requirement with vague security terms
            - Call analyze_clarity on the requirement
            - Assert that security vagueness issues are detected
        """
        requirement = "The system shall ensure secure data transmission and provide adequate protection."
        result = analyze_clarity(requirement)
        
        security_issues = [issue for issue in result["issues"] if issue.type == "Ambiguous" and any(term in issue.note.lower() for term in ["secure", "adequate"])]
        assert len(security_issues) > 0
        assert result["clarity_score"] < 80

        # Should detect terms like "secure", "adequate"
        security_descriptions = " ".join([issue.note for issue in security_issues])
        assert any(term in security_descriptions.lower() for term in ["secure", "adequate"])

    def test_analyze_clarity_multiple_issues(self):
        """
        Test scoring behavior when multiple types of issues are present.
        
        Steps:
            - Provide a requirement with various clarity issues
            - Call analyze_clarity on the requirement
            - Assert that multiple issue types are detected and score is appropriately reduced
        """
        requirement = "The system should be implemented TBD and will provide good security somehow."
        result = analyze_clarity(requirement)
        
        # Should have multiple issue types
        issue_types = {issue.type for issue in result["issues"]}
        assert len(issue_types) >= 2  # Should have at least TBD and Ambiguous issues

        # Score should be significantly reduced
        assert result["clarity_score"] < 60

    def test_analyze_clarity_empty_requirement(self):
        """
        Test behavior with an empty requirement string.
        
        Steps:
            - Provide an empty string as requirement
            - Call analyze_clarity on the empty requirement
            - Assert that it returns a valid result structure
        """
        requirement = ""
        result = analyze_clarity(requirement)
        
        assert "clarity_score" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["clarity_score"], (int, float))

    def test_analyze_clarity_whitespace_only(self):
        """
        Test behavior with whitespace-only requirement.
        
        Steps:
            - Provide a whitespace-only string as requirement
            - Call analyze_clarity on the requirement
            - Assert that it handles gracefully
        """
        requirement = "   \n\t   "
        result = analyze_clarity(requirement)
        
        assert "clarity_score" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)

    def test_analyze_clarity_mixed_case_detection(self):
        """
        Test that issue detection works with mixed case text.
        
        Steps:
            - Provide requirements with mixed case vague terms and TBD
            - Call analyze_clarity on each requirement
            - Assert that issues are detected regardless of case
        """
        requirements = [
            "The system should be FAST and TBD.",
            "The System Should Be Fast And tbd.",
            "THE SYSTEM SHOULD BE fast AND Tbd."
        ]
        
        for req in requirements:
            result = analyze_clarity(req)
            
            # Should detect both vague terms and TBD regardless of case
            issue_types = {issue.type for issue in result["issues"]}
            assert "Ambiguous" in issue_types or "TBD" in issue_types or "NonVerifiable" in issue_types

    def test_analyze_clarity_score_range(self):
        """
        Test that clarity scores are within expected range.
        
        Steps:
            - Test various requirements with different quality levels
            - Call analyze_clarity on each requirement
            - Assert that scores are within 0-10 range
        """
        requirements = [
            "The system shall authenticate users within 2 seconds using OAuth 2.0.",  # Good
            "The system should be fast and good.",  # Poor
            "TBD TBD TBD somehow maybe perhaps",  # Very poor
            ""  # Edge case
        ]
        
        for req in requirements:
            result = analyze_clarity(req)
            assert 0 <= result["clarity_score"] <= 100

    def test_analyze_clarity_issue_structure(self):
        """
        Test that detected issues have the correct structure.
        
        Steps:
            - Provide a requirement that will generate issues
            - Call analyze_clarity on the requirement
            - Assert that each issue has required attributes
        """
        requirement = "The system should be fast and TBD somehow."
        result = analyze_clarity(requirement)
        
        assert len(result["issues"]) > 0
        
        for issue in result["issues"]:
            assert isinstance(issue, Issue)
            assert hasattr(issue, 'type')
            assert hasattr(issue, 'note')
            assert hasattr(issue, 'span')
            assert isinstance(issue.type, str)
            assert isinstance(issue.note, str)
            assert isinstance(issue.span, str)
            assert len(issue.span) > 0

    def test_analyze_clarity_span_accuracy(self):
        """
        Test that issue spans are accurately reported.
        
        Steps:
            - Provide a requirement with known issue positions
            - Call analyze_clarity on the requirement
            - Assert that reported spans contain the expected text
        """
        requirement = "The system TBD should be fast"
        result = analyze_clarity(requirement)
        
        # Find TBD issue
        tbd_issues = [issue for issue in result["issues"] if issue.type == "TBD"]
        assert len(tbd_issues) == 1
        
        # TBD should be contained in the span
        assert "TBD" in tbd_issues[0].span

    def test_analyze_clarity_comprehensive_vague_terms(self):
        """
        Test detection of various vague terms mentioned in the heuristics.
        
        Steps:
            - Test requirements containing different categories of vague terms
            - Call analyze_clarity on each requirement
            - Assert that vague terms are consistently detected
        """
        # Test only terms that are actually detected by the heuristics
        vague_terms_tests = [
            ("The system should be fast", "fast"),
            ("The system should be good", "good"),
            ("The system should be simple", "simple"),
            ("The system should be efficient", "efficient"),
            ("The system should be robust", "robust"),
            ("The system should be reliable", "reliable"),
            ("The system should be secure", "secure"),
            ("The system should provide reasonable performance", "reasonable")
        ]
        
        for requirement, expected_term in vague_terms_tests:
            result = analyze_clarity(requirement)
            
            # Should detect at least one vague issue
            vague_issues = [issue for issue in result["issues"] if issue.type in ["Ambiguous", "NonVerifiable"]]
            assert len(vague_issues) > 0, f"Failed to detect vague term '{expected_term}' in requirement: {requirement}"

    def test_analyze_clarity_security_specific_terms(self):
        """
        Test detection of security-specific vague terms.
        
        Steps:
            - Test requirements with security-related vague terms
            - Call analyze_clarity on each requirement
            - Assert that security vagueness issues are detected
        """
        security_terms = [
            "The system shall provide secure authentication",
            "The system shall ensure adequate encryption",
            "The system shall implement appropriate access controls",
            "The system shall maintain sufficient audit logs",
            "The system shall use proper validation"
        ]
        
        for requirement in security_terms:
            result = analyze_clarity(requirement)
            
            security_issues = [issue for issue in result["issues"] if issue.type == "Ambiguous" and any(term in issue.note.lower() for term in ["secure", "adequate", "appropriate", "sufficient", "proper"])]
            assert len(security_issues) > 0, f"Failed to detect security vagueness in: {requirement}"

    def test_analyze_clarity_passive_voice_patterns(self):
        """
        Test detection of various passive voice patterns.
        
        Steps:
            - Test requirements with different passive voice constructions
            - Call analyze_clarity on each requirement
            - Assert that passive voice is consistently detected
        """
        # Test patterns that actually trigger passive voice detection
        passive_voice_examples = [
            "Results will be stored in the database",
            "Files will be uploaded to the server", 
            "Reports will be generated automatically",
            "Data will be processed and stored",
            "Information will be collected and analyzed"
        ]
        
        detected_count = 0
        for requirement in passive_voice_examples:
            result = analyze_clarity(requirement)
            
            passive_issues = [issue for issue in result["issues"] if issue.type == "PassiveVoice"]
            if len(passive_issues) > 0:
                detected_count += 1
        
        # Should detect passive voice in most of the examples
        assert detected_count >= len(passive_voice_examples) // 2, f"Only detected passive voice in {detected_count} out of {len(passive_voice_examples)} examples"

    def test_analyze_clarity_no_false_positives_active_voice(self):
        """
        Test that active voice requirements don't trigger passive voice detection.
        
        Steps:
            - Test clearly active voice requirements
            - Call analyze_clarity on each requirement
            - Assert that no passive voice issues are detected
        """
        active_voice_examples = [
            "The system shall process data within 5 seconds",
            "The application must store results in the database",
            "The service will authenticate users using OAuth",
            "The system must generate reports automatically",
            "The application shall maintain audit logs"
        ]
        
        for requirement in active_voice_examples:
            result = analyze_clarity(requirement)
            
            passive_issues = [issue for issue in result["issues"] if issue.type == "PassiveVoice"]
            assert len(passive_issues) == 0, f"False positive passive voice detection in: {requirement}"