import pytest

import numpy as np
import networkx as nx

from emb.classifiers import features

# =========== TESTS ===========
class TestFeatureHelpers:
    def test_get_normalized_feature(self):
        nums = [1, 1, 1]
        dens_cut_nums = [1, 1, 1]
        scaling = lambda x: x

        expected = [0.5, 0.5, 0.5]
        actual = features._get_normalized_feature([nums, dens_cut_nums], scaling=scaling)

        assert np.equal(actual, expected).all()

    def test_get_normalized_feature_scaled(self):
        nums = [1, 1, 1]
        dens_cut_nums = [1, 1, 1]

        expected = [0, 0, 0,]
        actual = features._get_normalized_feature([nums, dens_cut_nums])

        assert np.equal(actual, expected).all()

    def test_get_normalized_feature_zero_scaled(self):
        nums = [0, 0, 0]
        dens_cut_nums = [1, 1, 1]

        expected = [-1, -1, -1]
        actual = features._get_normalized_feature([nums, dens_cut_nums])

        assert np.equal(actual, expected).all()

    def test_get_normalized_feature_zde(self):
        with pytest.raises(ZeroDivisionError):
            nums = [1, 1, 1]
            dens_cut_nums = [-1, -1, -1]

            features._get_normalized_feature([nums, dens_cut_nums])

class TestFeaturesDegrees:
    def test_degree_product(self):
        G = nx.path_graph(10)
        edge = (0,1)

        expected = 2
        actual = features.degree_product(G, edge)

        assert actual == expected

    def test_degree_product_err(self):
        with pytest.raises(nx.exception.NetworkXError):
            G = nx.path_graph(10)
            edge = (0,37)

            features.degree_product(G, edge, training=False)

    def test_degree_product_catch_err(self):
        G = nx.path_graph(10)
        edge = (0,37)

        expected = 0
        actual = features.degree_product(G, edge, training=True)

        assert actual == expected

class TestFeaturesDistances:
    def test_inverse_vector_distance(self):
        vectors = {
            0: [0, 0],
            1: [3, 4]
        }
        edge = (0, 1)

        expected = 1 / 5
        actual = features.inverse_vector_distance(vectors, edge)

        assert actual == expected

    def test_inverse_vector_distance_equal(self):
        vectors = {
            0: [0, 0],
            1: [0, 0]
        }
        edge = (0, 1)

        expected = 1 / 1e-32
        actual = features.inverse_vector_distance(vectors, edge)

        assert actual == expected