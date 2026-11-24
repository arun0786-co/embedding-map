import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas.labels import stem  # noqa: E402


def test_plural_and_singular_share_a_stem():
    for a, b in [("galaxies", "galaxy"), ("fields", "field"), ("black holes", "black hole")]:
        assert [stem(w) for w in a.split()] == [stem(w) for w in b.split()]
    assert stem("glass") == "glass" and stem("gas") == "gas"
