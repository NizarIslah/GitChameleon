import pytest
import spacy
from spacy.pipeline.span_ruler import SpanRuler
from dataset.solutions.sample_93 import remove_pattern_by_id


@pytest.fixture
def nlp():
    """Create a small spaCy pipeline for testing."""
    return spacy.blank("en")


@pytest.fixture
def span_ruler(nlp):
    """Create a span ruler with some test patterns."""
    ruler = nlp.add_pipe("span_ruler", name="test_ruler")
    patterns = [
        {"id": "pattern1", "pattern": [{"LOWER": "apple"}], "label": "FRUIT"},
        {"id": "pattern2", "pattern": [{"LOWER": "orange"}], "label": "FRUIT"},
        {"id": "pattern3", "pattern": [{"LOWER": "car"}], "label": "VEHICLE"}
    ]
    ruler.add_patterns(patterns)
    return ruler


def test_remove_pattern_by_id(nlp, span_ruler):
    """Test that a pattern can be removed by ID."""
    # Get the span ruler from the pipeline
    ruler = nlp.get_pipe("test_ruler")
    
    # Verify we have 3 patterns initially
    assert len(ruler.patterns) == 3
    
    # Remove a pattern by ID
    remove_pattern_by_id(ruler, "pattern2")
    
    # Verify the pattern was removed
    assert len(ruler.patterns) == 2
    
    # Verify the correct pattern was removed
    pattern_ids = [p["id"] for p in ruler.patterns]
    assert "pattern1" in pattern_ids
    assert "pattern2" not in pattern_ids
    assert "pattern3" in pattern_ids


def test_remove_nonexistent_pattern(nlp, span_ruler):
    """Test removing a pattern that doesn't exist."""
    # Get the span ruler from the pipeline
    ruler = nlp.get_pipe("test_ruler")
    
    # Verify we have 3 patterns initially
    assert len(ruler.patterns) == 3
    
    # Attempt to remove a non-existent pattern
    # This should not raise an error
    remove_pattern_by_id(ruler, "nonexistent_pattern")
    
    # Verify no patterns were removed
    assert len(ruler.patterns) == 3