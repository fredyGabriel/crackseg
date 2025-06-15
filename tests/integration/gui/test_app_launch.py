"""Integration tests for the main Streamlit application launch."""

from streamlit.testing.v1 import AppTest

# The path to the app file should be relative to the project root,
# where pytest is executed.
APP_FILE = "scripts/gui/app.py"


class TestAppLaunch:
    """Test suite for application launch and basic rendering."""

    def test_app_launches_without_errors(self):
        """
        Test that the application runs, initializes session state,
        and renders without raising any exceptions.
        """
        at = AppTest.from_file(APP_FILE).run(timeout=10)
        # A successful run should not have an exception.
        assert not at.exception

    def test_initial_page_render(self):
        """
        Test that the default page renders with the correct title and
        initial navigation breadcrumbs.
        """
        at = AppTest.from_file(APP_FILE).run(timeout=10)

        # Check for the main title (from st.set_page_config)
        # Note: AppTest doesn't directly expose page_title, so we check
        # rendered content that depends on it.

        # Check for the navigation breadcrumb by searching all markdown
        # elements
        nav_breadcrumb = None
        for md in at.markdown:
            if "Navigation:" in md.value:
                nav_breadcrumb = md.value
                break

        assert nav_breadcrumb is not None, "Navigation breadcrumb not found"

        # The default page is 'page_config', so its breadcrumb should be
        # present
        assert "Config" in nav_breadcrumb
