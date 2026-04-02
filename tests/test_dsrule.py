import numpy as np

from src.DSRule import DSRule


class TestDSRule:
    def test_caption(self):
        r = DSRule(lambda x: x > 3, "x > 3")
        assert str(r) == "x > 3"

    def test_callable_true(self):
        r = DSRule(lambda x: x[0] > 0, "X[0] > 0")
        assert r(np.array([1.0, 2.0])) == True

    def test_callable_false(self):
        r = DSRule(lambda x: x[0] > 0, "X[0] > 0")
        assert r(np.array([-1.0, 2.0])) == False

    def test_empty_caption(self):
        r = DSRule(lambda x: True)
        assert str(r) == ""

    def test_range_rule(self):
        r = DSRule(lambda x: 0 <= x[0] < 5, "0 <= X[0] < 5")
        assert r(np.array([3.0]))
        assert not r(np.array([5.0]))
        assert not r(np.array([-1.0]))
