import pytest

import numpy as np

from EMB import utils


class TestFunctions:
    def test_dict2list(self):
        d = {
            1: [0, 0],
            37: [1, 2],
        }

        expected_list = [[0, 0], [1, 2]]
        expected_rel = {1: 0, 37: 1}

        actual_list, actual_rel = utils.dict2arr(d)

        assert np.equal(actual_list, expected_list).all() and actual_rel == expected_rel

    def test_dict2arr(self):
        d = {
            1: [0, 0],
            37: [1, 2],
        }

        expected_arr = np.array([[0, 0], [1, 2]])
        expected_rel = {1: 0, 37: 1}

        actual_arr, actual_rel = utils.dict2arr(d)

        assert np.equal(actual_arr, expected_arr).all() and (actual_rel == expected_rel)

    def test_list2dict(self):
        list_ = [4, 3, 1, 2]
        index_mapping = {0: 3, 1: 2, 2: 0, 3: 1}

        expected = {0: 1, 1: 2, 2: 3, 3: 4}
        actual = utils.list2dict(list_, index_mapping)

        assert actual == expected

    def test_cutkey(self):
        d = {
            1: 1,
            2: 2,
            3: "a",
        }

        expected = {1: 1, 3: "a"}
        actual = utils.cutkey(d, 2)

        assert actual == expected

    def test_cutkey_nested(self):
        d = {
            1: 1,
            2: 2,
            3: {1: 1, 2: 2, 3: "a"},
        }

        expected = {1: 1, 3: {1: 1, 2: 2, 3: "a"}}
        actual = utils.cutkey(d, 2)

        assert actual == expected

    def test_cutkey_multiple(self):
        d = {
            1: 1,
            2: 2,
            3: "a",
        }

        expected = {1: 1}
        actual = utils.cutkey(d, 2, 3)

        assert actual == expected

    def test_inverse_map(self):
        mapping = {1: 37, 2: 5, 3: 1, 4: 90}
        expected = {37: 1, 5: 2, 1: 3, 90: 4}
        actual = utils.inverse_map(mapping)

        assert actual == expected

    def test_inverse_map_asserterror(self):
        with pytest.raises(AssertionError):
            mapping = {1: 37, 2: 0, 3: 37}
            utils.inverse_map(mapping)
