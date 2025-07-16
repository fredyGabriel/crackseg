from gui.utils.gui_config import PAGE_CONFIG


def test_page_config_exists() -> None:
    """
    Test that PAGE_CONFIG is defined and is a dictionary.
    """
    assert isinstance(PAGE_CONFIG, dict)
    assert len(PAGE_CONFIG) > 0


def test_page_config_keys() -> None:
    """Test that all expected pages are present in PAGE_CONFIG."""
    expected_pages = {
        "Home",
        "Config",
        "Advanced Config",
        "Architecture",
        "Train",
        "Results",
    }
    assert set(PAGE_CONFIG.keys()) == expected_pages


def test_page_config_structure() -> None:
    """Test the structure of each page configuration."""
    for _page, config in PAGE_CONFIG.items():
        assert isinstance(config, dict)
        required_keys = {"title", "icon", "description", "requires"}
        assert required_keys.issubset(config.keys())
        assert isinstance(config["title"], str)
        assert isinstance(config["icon"], str)
        assert isinstance(config["description"], str)
        assert isinstance(config["requires"], list)
        for req in config["requires"]:
            assert isinstance(req, str)


def test_page_requires() -> None:
    """Test specific requires for pages that need them."""
    assert PAGE_CONFIG["Architecture"]["requires"] == ["config_loaded"]
    assert PAGE_CONFIG["Train"]["requires"] == [
        "config_loaded",
        "run_directory",
    ]
    assert PAGE_CONFIG["Results"]["requires"] == ["run_directory"]
    assert PAGE_CONFIG["Home"]["requires"] == []
