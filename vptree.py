""" This module contains an implementation of a Vantage Point-tree (VP-tree)."""
from typing import Callable, List, Type
import numpy as np
import copy


class VPTree:

    """ VP-Tree data structure for efficient nearest neighbor search.

    The VP-tree is a data structure for efficient nearest neighbor
    searching and finds the nearest neighbor in O(log n)
    complexity given a tree constructed of n data points. Construction
    complexity is O(n log n).

    Parameters
    ----------
    points : Iterable
        Construction points.
    dist_fn : Callable
        Function taking to point instances as arguments and returning
        the distance between them.
    leaf_size : int
        Minimum number of points in leaves (IGNORED).
    """

    # tolerance fopr safe comparissons given numpy's float point rounding issue
    tolerance = 1e-6

    # Maximum difference allowed between two children nodes'cardinality.
    # It is ALWAYS larger than one! `max_diff >= 2`
    # It dictates how unbalanced the tree can be when adding nodes a posteriori
    max_diff = 100

    # Dictionary to store all the points according to their ids (int:Point)
    points = {}

    # Function that computes the desired distance metric
    dist_fn = None

    # Dictionary mapping a point id to the nodes in which it is a vp-point (int:VPTree)
    vp_id_to_node = {}


    ### FOR DEBUG:
    # timestamp = 0
    ##

    # "Public" constructor. Use this!
    @classmethod
    def construct_tree(cls, point_list : List[float], distance_metric : Callable):
        if not len(point_list):
            raise ValueError('Points can not be empty.')

        if not distance_metric:
            raise ValueError('A distance metric is mandatory.')

        VPTree.dist_fn = distance_metric
        VPTree.points = {id:point for id,point in enumerate(point_list)}

        return VPTree(list(range(len(point_list))), None)        


    # "Private" constructor used for recursion only. Use `construct_tree` instead!
    def __init__(self, point_ids : List[int], parent):
        
        ### FOR DEBUG:
        # self.tstamp = VPTree.timestamp
        # VPTree.timestamp = VPTree.timestamp + 1
        ###

        self.parent = parent
        self.left = None
        self.right = None

        self.left_min = np.inf
        self.left_max = 0.0

        self.right_min = np.inf
        self.right_max = 0.0

        if not parent:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        # Vantage point is point furthest from parent vp.
        self.vp_id = point_ids[0]
        VPTree.vp_id_to_node[self.vp_id] = self

        # self.node_point_ids = np.delete(point_ids, 0, axis=0)
        self.node_point_ids = copy.deepcopy(point_ids[1:])

        if len(self.node_point_ids) is 0:
            return

        # Choose division boundary at median of distances.
        distances = [VPTree.dist_fn(VPTree.points[self.vp_id], VPTree.points[p]) for p in self.node_point_ids]
        self.median = np.median(distances)

        left_points = []
        right_points = []
        for point_id, distance in zip(self.node_point_ids, distances):

            if distance >= (self.median - VPTree.tolerance):
                self.right_min = min(distance, self.right_min)

                if distance > (self.right_max + VPTree.tolerance):
                    self.right_max = distance
                    right_points.insert(0, point_id) # put furthest first
                else:
                    right_points.append(point_id)

            else:
                self.left_min = min(distance, self.left_min)

                if distance > (self.left_max + VPTree.tolerance):
                    self.left_max = distance
                    left_points.insert(0, point_id) # put furthest first
                else:
                    left_points.append(point_id)

        if len(left_points) > 0:
            self.left = VPTree(left_points, self)

        if len(right_points) > 0:
            self.right = VPTree(right_points, self)


    def _is_leaf(self):
        return (self.left is None) and (self.right is None)


    def add_point(self, point):
        """ Adds a new point into the tree.
        
        Parameters
        ----------
        point : Any
            Point.
        """

        point = np.array(point)
        if not np.any(point):
            raise ValueError('Points can not be empty.')

        new_point_id = len(VPTree.points)
        VPTree.points[new_point_id] = point

        self.__traverse(new_point_id)

    def __traverse(self, new_point_id):
        """ Traverses the tree and creates a new node for the added point
        
        Parameters
        ----------
        point_is : int
            Point id.
        """

        # If the current node is a leaf, update bounds and create a new node
        if self._is_leaf():
            self.__init__([new_point_id, self.vp_id], self.parent)
            return
                        
        # Get number of points in the left node
        n_points_left = 0            
        if self.left:
            n_points_left = len(self.left.node_point_ids)

        # Get number of points in the right node
        n_points_right = 0
        if self.right:
            n_points_right = len(self.right.node_point_ids)
            
        #  If the subtree is too unbalanced (according to `VPTree.max_diff`), resconstruct it
        if abs(n_points_left - n_points_right) > VPTree.max_diff:

            if self.parent is None:
                all_node_ids = self.node_point_ids
                all_node_ids.append(new_point_id)
                all_node_ids.insert(0, self.vp_id)

            else:
                distance_vp = VPTree.dist_fn(VPTree.points[self.vp_id], VPTree.points[self.parent.vp_id])
                distance_new_point = VPTree.dist_fn(VPTree.points[new_point_id], VPTree.points[self.parent.vp_id])
                
                all_node_ids = self.node_point_ids
                if(distance_new_point > (distance_vp + VPTree.tolerance)):
                    all_node_ids.append(self.vp_id)
                    all_node_ids.insert(0, new_point_id)
                else:
                    all_node_ids.append(new_point_id)
                    all_node_ids.insert(0, self.vp_id)

            self.__init__(all_node_ids, self.parent)
            return

        # In this case, update the current node and keep traversing
        distance = VPTree.dist_fn(VPTree.points[self.vp_id], VPTree.points[new_point_id])
        if distance >= (self.median - VPTree.tolerance):

            if distance > (self.right_max + VPTree.tolerance):
                self.right_max = distance

            if distance < (self.right_min - VPTree.tolerance):
                self.right_min = distance
                                     

            if self.right is None:
                self.right = VPTree([new_point_id], self)
            else:
                self.node_point_ids.append(new_point_id)
                self.right.__traverse(new_point_id)

        else:
            if distance > (self.left_max + VPTree.tolerance):
                self.left_max = distance

            if distance < (self.left_min - VPTree.tolerance):
                self.left_min = distance


            if self.left is None:
                self.left = VPTree([new_point_id], self)
            else:
                self.node_point_ids.append(new_point_id)
                self.left.__traverse(new_point_id)
        

    def get_nearest_neighbor(self, query):
        """ Get single nearest neighbor.
        
        Parameters
        ----------
        query : Any
            Query point.

        Returns
        -------
        Any
            Single nearest neighbor.
        """
        return self.get_n_nearest_neighbors(query, n_neighbors=1)[0]

    def get_n_nearest_neighbors(self, query, n_neighbors):
        """ Get `n_neighbors` nearest neigbors to `query`
        
        Parameters
        ----------
        query : Any
            Query point.
        n_neighbors : int
            Number of neighbors to fetch.

        Returns
        -------
        list
            List of `n_neighbors` nearest neighbors.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be strictly positive integer')
            
        neighbors = _AutoSortingList(max_size=n_neighbors)
        nodes_to_visit = [(self, 0.0)]

        furthest_d = np.inf

        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > (furthest_d + VPTree.tolerance):
                continue
            
            d = VPTree.dist_fn(query, VPTree.points[node.vp_id])
            if d < (furthest_d - VPTree.tolerance):
                neighbors.append((d, VPTree.points[node.vp_id]))
                
                # Update the furtherst distance after having found at least `n_neighbors`
                if len(neighbors) >= n_neighbors:
                    furthest_d, _ = neighbors[-1]

            if node._is_leaf():
                continue


            if (node.left_min - VPTree.tolerance) <= d <= (node.left_max + VPTree.tolerance):
                nodes_to_visit.insert(0, (node.left, 0.0))

            elif (node.left_min - furthest_d - VPTree.tolerance) <= d <= (node.left_max + furthest_d + VPTree.tolerance):
                nodes_to_visit.append((node.left,
                                       node.left_min - d if d < (node.left_min - VPTree.tolerance)
                                       else d - node.left_max))


            if (node.right_min - VPTree.tolerance) <= d <= (node.right_max + VPTree.tolerance):
                nodes_to_visit.insert(0, (node.right, 0.0))

            elif (node.right_min - furthest_d - VPTree.tolerance) <= d <= (node.right_max + furthest_d + VPTree.tolerance):
                nodes_to_visit.append((node.right,
                                       node.right_min - d if (d < node.right_min - VPTree.tolerance)
                                       else d - node.right_max))

        return list(neighbors)

    def get_all_in_range(self, query, max_distance):
        """ Find all neighbours within `max_distance`.

        Parameters
        ----------
        query : Any
            Query point.
        max_distance : float
            Threshold distance for query.

        Returns
        -------
        neighbors : list
            List of points within `max_distance`.

        Notes
        -----
        Returned neighbors are not sorted according to distance.
        """
        neighbors = list()
        nodes_to_visit = [(self, 0)]

        while len(nodes_to_visit) > 0:
            node, d0 = nodes_to_visit.pop(0)
            if node is None or d0 > (max_distance + VPTree.tolerance):
                continue

            d = VPTree.dist_fn(query, VPTree.points[node.vp_id])
            if d < (max_distance - VPTree.tolerance):
                neighbors.append((d, VPTree.points[node.vp_id]))

            if node._is_leaf():
                continue


            if (node.left_min - VPTree.tolerance) <= d <= (node.left_max + VPTree.tolerance):
                nodes_to_visit.insert(0, (node.left, 0))

            elif (node.left_min - max_distance - VPTree.tolerance) <= d <= (node.left_max + max_distance + VPTree.tolerance):
                nodes_to_visit.append((node.left,
                                       node.left_min - d if (d < node.left_min - VPTree.tolerance)
                                       else d - node.left_max))


            if (node.right_min - VPTree.tolerance) <= d <= (node.right_max + VPTree.tolerance):
                nodes_to_visit.insert(0, (node.right, 0))

            elif (node.right_min - max_distance - VPTree.tolerance) <= d <= (node.right_max + max_distance + VPTree.tolerance):
                nodes_to_visit.append((node.right,
                                       node.right_min - d if d < (node.right_min - VPTree.tolerance)
                                       else d - node.right_max))

        return neighbors

class _AutoSortingList(list):

    """ Simple auto-sorting list.

    Inefficient for large sizes since the queue is sorted at
    each push.

    Parameters
    ---------
    size : int, optional
        Max queue size.
    """

    def __init__(self, max_size=None, *args):
        super(_AutoSortingList, self).__init__(*args)
        self.max_size = max_size

    def append(self, item):
        """ Append `item` and sort.

        Parameters
        ----------
        item : Any
            Input item.
        """
        super(_AutoSortingList, self).append(item)
        self.sort(key=lambda x: x[0])
        if self.max_size is not None and len(self) > self.max_size:
            self.pop()

