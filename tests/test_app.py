"""End-to-end smoke test for the four Streamlit views."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppFlowTests(unittest.TestCase):
    def test_complete_user_journey_has_no_exceptions(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Find your\nsomewhere.")

        # Repeatedly choose the left card, exercising model refits and active
        # pair selection on every rerun.
        for _ in range(4):
            app.button[0].click().run()
            self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["comparisons"]), 4)

        app.radio[0].set_value("My Taste").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Your taste, decoded.")
        recommendation_headings = [
            element.value for element in app.markdown if element.value.startswith("### ")
        ]
        self.assertEqual(len(recommendation_headings), 6)

        app.radio[0].set_value("Group Trip").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Find the group’s happy place.")
        self.assertTrue(any("at least two" in element.value for element in app.warning))

        app.multiselect[0].set_value(["Maya · culture + food", "Theo · wild outdoors"]).run()
        self.assertFalse(app.exception)
        group_headings = [element.value for element in app.markdown if element.value.startswith("### ")]
        self.assertEqual(len(group_headings), 6)

        app.radio[0].set_value("How It Works").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Small choices. Useful intelligence.")


if __name__ == "__main__":
    unittest.main()
