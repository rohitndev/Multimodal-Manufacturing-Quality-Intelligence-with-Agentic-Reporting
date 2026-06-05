from src.vision import VisionPipeline
from src.vision.detector import DefectDetector
from src.vision.severity import SeverityClassifier


def test_detector_returns_list(synthetic_image):
    det = DefectDetector()
    out = det.detect(synthetic_image)
    assert isinstance(out, list)
    if out:
        first = out[0]
        assert first.defect_type
        assert 0 <= first.confidence <= 1
        assert len(first.bbox) == 4


def test_severity_classifier_levels(synthetic_image):
    sev = SeverityClassifier()
    result = sev.classify(synthetic_image, bbox=[100, 230, 540, 250])
    assert result.level in {"Critical", "Major", "Minor"}
    assert 0 <= result.score <= 1
    assert set(result.probabilities) == {"Critical", "Major", "Minor"}


def test_pipeline_end_to_end(synthetic_image, tmp_path):
    pipe = VisionPipeline()
    overlay = tmp_path / "overlay.png"
    result = pipe.run(synthetic_image, image_id="unit-test", overlay_output=str(overlay))
    assert result.image_id == "unit-test"
    assert isinstance(result.findings, list)
