import importlib.util
from pathlib import Path


def load_comprehensive():
    """Load comprehensive_tutorial from package if possible, else from file."""
    try:
        from data_toolkit import comprehensive_tutorial as ct
        return ct
    except Exception:
        p = Path("src/data_toolkit/comprehensive_tutorial.py").resolve()
        spec = importlib.util.spec_from_file_location("comprehensive_tutorial_local", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def test_image_recognition_has_comprehensive_content():
    ct = load_comprehensive()
    from data_toolkit.tabs import tutorial_sidebar as tb

    short = tb.TUTORIALS.get("image_recognition", "")
    long = ct.get_tutorial("image_recognition")

    assert "Image Recognition" in long
    # The comprehensive content should be longer than the short sidebar summary
    assert len(long) > len(short)


def test_sidebar_mapping_for_statistical():
    ct = load_comprehensive()
    # The sidebar key 'statistical' should map to comprehensive 'descriptive_stats'
    assert 'descriptive_stats' in ct.get_all_topics()
    desc = ct.get_tutorial('descriptive_stats')
    assert 'Descriptive Statistics' in desc or 'mean' in desc


def test_pca_mapping_uses_comprehensive():
    ct = load_comprehensive()
    from data_toolkit.tabs import tutorial_sidebar as tb

    # Our mapping should point 'pca' -> 'pca_analysis' and the comprehensive
    # tutorial should contain PCA-specific guidance (Explained Variance, Scree Plot).
    assert 'pca_analysis' in ct.get_all_topics()
    long = ct.get_tutorial('pca_analysis')
    assert 'Explained Variance' in long or 'Scree Plot' in long or 'Explained Variance Ratio' in long


