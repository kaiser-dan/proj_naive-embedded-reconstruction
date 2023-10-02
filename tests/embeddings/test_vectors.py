import numpy as np

from EMB.embeddings import vectors


# =========== TESTS ===========
class TestVectorHelpers:
    def test_get_center_of_mass(self):
        vectors_ = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        expected = [1, 1, 1]
        actual = vectors._get_center_of_mass(vectors_)

        assert all(np.equal(actual, expected))

    def test_get_center_of_mass_trivial(self):
        vectors_ = [
            [1, 1, 1],
        ]

        expected = [1, 1, 1]
        actual = vectors._get_center_of_mass(vectors_)

        assert all(np.equal(actual, expected))

    def test_get_norm(self):
        vectors_ = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        expected = np.sqrt(3) + np.sqrt(12)
        actual = vectors._get_component_norm(vectors_)

        assert actual == expected

    def test_get_norm_trivial(self):
        vectors_ = [
            [1, 1, 1],
        ]

        expected = np.sqrt(3)
        actual = vectors._get_component_norm(vectors_)

        assert actual == expected


# TODO: Add tests with non-default node2id kwarg
class TestVectorNormalization:
    vectors_ = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    components_single = {(0, 1, 2)}
    components_double = {(0, 1), (2,)}

    def test_align_vectors(self):
        expected = {0: [-1, -1, -1], 1: [0, 0, 0], 2: [1, 1, 1]}
        actual = vectors.align_vectors(self.vectors_, self.components_single)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2

    def test_align_vectors_twocomps(self):
        expected = {0: [-0.5, -0.5, -0.5], 1: [0.5, 0.5, 0.5], 2: [0, 0, 0]}
        actual = vectors.align_vectors(self.vectors_, self.components_double)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2

    def test_scale_vectors(self):
        norm = np.sqrt(3) + np.sqrt(12)

        expected = {
            0: [0, 0, 0],
            1: [1 / norm, 1 / norm, 1 / norm],
            2: [2 / norm, 2 / norm, 2 / norm],
        }
        actual = vectors.scale_vectors(self.vectors_, self.components_single)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2

    def test_scale_vectors_twocomp(self):
        norm1 = np.sqrt(3)
        norm2 = np.sqrt(12)

        expected = {
            0: [0, 0, 0],
            1: [1 / norm1, 1 / norm1, 1 / norm1],
            2: [2 / norm2, 2 / norm2, 2 / norm2],
        }
        actual = vectors.scale_vectors(self.vectors_, self.components_double)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2

    def test_normalize_vectors(self):
        norm = 2 * np.sqrt(3)
        expected = {
            0: [-1 / norm, -1 / norm, -1 / norm],
            1: [0, 0, 0],
            2: [1 / norm, 1 / norm, 1 / norm],
        }
        actual = vectors.normalize_vectors(self.vectors_, self.components_single)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2

    def test_normalize_vectors_twocomp(self):
        norm = 2 * np.sqrt(0.75)
        expected = {
            0: [-0.5 / norm, -0.5 / norm, -0.5 / norm],
            1: [0.5 / norm, 0.5 / norm, 0.5 / norm],
            2: [0, 0, 0],
        }
        actual = vectors.normalize_vectors(self.vectors_, self.components_double)

        v0 = all(np.equal(expected[0], actual[0]))
        v1 = all(np.equal(expected[1], actual[1]))
        v2 = all(np.equal(expected[2], actual[2]))

        assert v0 and v1 and v2
