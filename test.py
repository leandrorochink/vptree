
from operator import le
import unittest

from numpy.lib.polynomial import polyint
from vptree import VPTree
import numpy as np


class TestVPTree(unittest.TestCase):
    
    def test_single_nearest_neighbor(self):
        dim = 10
        query = [.5] * dim
        points, brute_force = brute_force_solution(20000, dim, query)
        tree, ids = VPTree.construct_tree(points, euclidean)

        nearest = tree.get_nearest_neighbor(query)
        bf_nearest = brute_force[0]

        self.assertEqual(nearest[0], bf_nearest[0])
        self.assertTrue(all(n == b for n, b in zip(nearest[1], bf_nearest[1])))

    def test_nearest_neighbors(self):
        dim = 10
        query = [.5] * dim
        points, brute_force = brute_force_solution(20000, dim, query)
        tree, ids = VPTree.construct_tree(points, euclidean)

        for k in (1, 10, len(points)):
            tree_nearest = tree.get_n_nearest_neighbors(query, k)
            brute_force_nearest = brute_force[:k]

            for nearest, bf_nearest in zip(tree_nearest, brute_force_nearest):    
                self.assertEqual(nearest[0], bf_nearest[0])
                self.assertTrue(all(n == b for n, b in zip(nearest[1],
                                                           bf_nearest[1])))

    def test_epsilon_search(self):
        dim = 10
        query = [.5] * dim
        points, brute_force = brute_force_solution(20000, dim, query)
        tree, ids = VPTree.construct_tree(points, euclidean)

        for eps in (-1, 0, 1, 2, 10):
            tree_nearest = sorted(tree.get_all_in_range(query, eps))
            brute_force_nearest = [point for point in brute_force if
                                   point[0] < eps]

            for nearest, bf_nearest in zip(tree_nearest, brute_force_nearest):
                self.assertEqual(nearest[0], bf_nearest[0])
                self.assertTrue(all(n == b for n, b in zip(nearest[1],
                                                           bf_nearest[1])))

    def test_empty_points_raises_valueerror(self):
        self.assertRaises(ValueError, VPTree.construct_tree, [], euclidean)

    def test_zero_neighbors_raises_valueerror(self):
        tree, ids = VPTree.construct_tree(np.array([1.0, 2.0, 3.0]), euclidean)
        self.assertRaises(ValueError, tree.get_n_nearest_neighbors, [1], 0)

    def test_adding_points(self):
        dim = 10
        query = [.5] * dim
        points, brute_force = brute_force_solution(20000, dim, query)
        tree, ids = VPTree.construct_tree(points[:10000], euclidean)

        for point in points[10000:]:
            tree.add_point(point)

        for k in (1, 10, len(points)):
            tree_nearest = tree.get_n_nearest_neighbors(query, k)
            brute_force_nearest = brute_force[:k]

            for nearest, bf_nearest in zip(tree_nearest, brute_force_nearest):    
                self.assertEqual(nearest[0], bf_nearest[0])
                self.assertTrue(all(n == b for n, b in zip(nearest[1],
                                                           bf_nearest[1])))


    def test_removing_points(self):
        dim = 10
        query = [.5] * dim
        points, brute_force = brute_force_solution(10000, dim, query)
        tree, ids = VPTree.construct_tree(points, euclidean)

        #add new extra points
        random_gen = np.random.default_rng(seed=2)
        aditional_points = 10 * random_gen.standard_normal(size=(5000, dim))
        extra_points_ids = []
        for point in aditional_points:
            extra_points_ids.append(tree.add_point(point))

        for point_id in extra_points_ids:
            tree.remove_point(point_id)

        for k in (1, 10, len(points)):
            tree_nearest = tree.get_n_nearest_neighbors(query, k)
            brute_force_nearest = brute_force[:k]
        
            for nearest, bf_nearest in zip(tree_nearest, brute_force_nearest):                
                self.assertEqual(nearest[0], bf_nearest[0])
                self.assertTrue(all(n == b for n, b in zip(nearest[1],
                                                            bf_nearest[1])))


def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


def brute_force_solution(n, dim, query, dist=euclidean):
    random_gen = np.random.default_rng(seed=1)
    points = 10 * random_gen.standard_normal(size=(n, dim))
    
    # points = np.random.randn(n, dim)
    
    brute_force = [(dist(query, point), point) for point in points]
    brute_force.sort()

    return points, brute_force


if __name__ == '__main__':
    unittest.main()