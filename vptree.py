""" This module contains an implementation of a Vantage Point-tree (VP-tree)."""
from typing import Any, Callable, List, Type, Dict, Tuple
import numpy as np
import copy

class VPTree:

    """ VP-Tree data structure for efficient nearest neighbor search.

    The VP-tree is a data structure for efficient nearest neighbor
    searching and finds the nearest neighbor in O(log n)
    complexity given a tree constructed of n data points. Construction
    complexity is O(n log n).
    """

    # Tolerance fopr safe comparissons given numpy's float point rounding issue
    tolerance = 1e-6

    # Next id available for new data points
    next_id = int(0)

    # Maximum difference allowed between two children nodes'cardinality.
    # It is ALWAYS larger than one! `max_diff >= 2`
    # It dictates how unbalanced the tree can be when adding nodes a posteriori
    max_diff = 100

    # Dictionary to store all the points according to their ids (int:Point)
    points = {}

    # Function that computes the desired distance metric
    dist_fn = None
    stored_distances = {}

    # Dictionary mapping a point id to the nodes in which it is a vp-point (int:VPTree)
    vp_id_to_node = {}

    @classmethod
    def __get_next_id(cls):        
        VPTree.next_id = VPTree.next_id + int(1)
        return VPTree.next_id - int(1)

    # "Public" constructor. Use this!
    @classmethod
    def construct_tree(cls, point_list : List[List[float]], distance_metric : Callable):
        """ Constructs a VP-Tree data structure for efficient nearest neighbor search.
        !!! USE THIS FACTORY METHOD INSTEAD OF THE CLASS CONSTRUCTOR !!!

        Parameters
        ----------
        point_list : Iterable
            Points to be partitoned.
        dist_fn : Callable
            Function taking point instances as arguments and returning
            a distance metric between them.

        Returns
        -------
        Any
            A vantage point tree object
        Dict [int,int]
            A dictonary containing the Id of the points in the tree.
            The ids are set following the order in which they are added, i.e., the ordering in 'point_list'
        """
        if not len(point_list):
            raise ValueError('Points can not be empty.')

        if not distance_metric:
            raise ValueError('A distance metric is mandatory.')

        point_ids = {}
        for i in range(len(point_list)):
            id = VPTree.__get_next_id()            
            point_ids[id] = id

        VPTree.vp_id_to_node = {}
        VPTree.dist_fn = distance_metric
        VPTree.points = {id:point for id,point in zip(list(point_ids.keys()), point_list)}

        initial_vp_id = next(iter(point_ids.keys()))
        return VPTree(point_ids, None, initial_vp_id), point_ids

    @staticmethod
    def distance_by_id(id_1, id_2):
        """ Get distance between points acoording to the given point ids.
        It uses the distance metric provided when calling the 'VPTree.construct_tree' method.
        
        Parameters
        ----------
        id_1 : int
            Point id.
        
        id_2 : int
            Point id.

        Returns
        -------
        float
            Distance between points with id_1 and id_2 acording to 'VPTree.dist_fn'
        """

        if (not isinstance(id_1, int)) or (not isinstance(id_2, int)):
            raise ValueError('The point id must be an integer')

        # print("In tree::::")
        # print("vp-tree:: ", list(VPTree.points.keys()), flush=True)

        if id_1 not in VPTree.points:
            raise ValueError('The point ids must exist. Tried id_1 = {}'.format(id_1))
        
        if id_2 not in VPTree.points:
            raise ValueError('The point ids must exist. Tried id_2 = {}'.format(id_2))

        if not (id_1, id_2) in VPTree.stored_distances:
            VPTree.stored_distances[(id_1, id_2)] = VPTree.dist_fn(VPTree.points[id_1], VPTree.points[id_2])
            VPTree.stored_distances[(id_2, id_1)] = VPTree.stored_distances[(id_1, id_2)]

        return VPTree.stored_distances[(id_1, id_2)]


    # "Private" constructor used for recursion only. Use `construct_tree` instead!
    def __init__(self, point_ids: Dict[int,int], parent: 'VPTree', furtherest_id: int):
        """ (Private) Constructor! Creates a VP-tree.
        
        Parameters
        ----------
        point_ids : Dict[int,int]
            Dictionary of point ids.
        
        parent : 'VPTree'
            Parent node in the tree.
            It is 'None' for the root node
        
        furtherest_id : int
            Id of the point to be used as the vantage point.

        Returns
        -------
        Any
            A vantage point tree node
        """

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


        self.node_point_ids = {}
        
        # Vantage point is point furthest from parent vp.
        self.vp_id = int(furtherest_id)        
        VPTree.vp_id_to_node[self.vp_id] = self

        self.node_point_ids = copy.deepcopy(point_ids)
        del self.node_point_ids[furtherest_id]

        if len(self.node_point_ids) is 0:
            return

        # Choose division boundary at median of distances.
        distances = [VPTree.distance_by_id(self.vp_id, int(p)) for p in list(self.node_point_ids.keys())]
        self.median = np.median(distances)

        left_points = {}
        vp_left = -1
        right_points = {}
        vp_right = -1
        for point_id, distance in zip(list(self.node_point_ids.keys()), distances):

            if distance >= (self.median - VPTree.tolerance):
                self.right_min = min(distance, self.right_min)

                if distance > (self.right_max + VPTree.tolerance):
                    self.right_max = distance
                    vp_right = int(point_id)

                right_points[point_id] = point_id
                
            else:
                self.left_min = min(distance, self.left_min)

                if distance > (self.left_max + VPTree.tolerance):
                    self.left_max = distance
                    vp_left = int(point_id)
                    
                left_points[point_id] = point_id                

        if len(left_points) > 0:
            self.left = None
            self.left = VPTree(left_points, self, vp_left)

        if len(right_points) > 0:
            self.right = None
            self.right = VPTree(right_points, self, vp_right)


    def _is_leaf(self):
        """ Checks whether the node is a leaf
        
        Returns
        -------
        bool
            True if the node if a leaf and False otherwise.
        """
        return (self.left is None) and (self.right is None)

   
    def add_point(self, point):
        """ Adds a new point into the tree.
        
        Parameters
        ----------
        point : Any
            Point.

        Returns
        -------
        int
            Id of the point added to the VP-tree.
        """

        point = np.array(point)
        if not np.any(point):
            raise ValueError('Points can not be empty.')

        new_point_id = VPTree.__get_next_id()
        VPTree.points[new_point_id] = point

        self.__traverse(new_point_id)

        return new_point_id
 

    def __traverse(self, new_point_id):
        """ Traverses the tree and creates a new node for the added point
        
        Parameters
        ----------
        new_point_id : int
            Point id.
        """

        # If the current node is a leaf, update bounds and create a new node
        if self._is_leaf():
            self.__init__({new_point_id:new_point_id, self.vp_id:self.vp_id}, self.parent, self.vp_id)
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

            vp = copy.deepcopy(self.vp_id)
            all_node_ids = copy.deepcopy(self.node_point_ids)
            
            if not self.parent is None:                
                distance_vp = VPTree.distance_by_id(self.vp_id, self.parent.vp_id)
                distance_new_point = VPTree.distance_by_id(new_point_id, self.parent.vp_id)
                
                if(distance_new_point > (distance_vp + VPTree.tolerance)):
                    vp = new_point_id
            
            all_node_ids[new_point_id] = new_point_id
            all_node_ids[self.vp_id] = self.vp_id

            self.__init__(all_node_ids, self.parent, vp)
            
            return

        # In this case, update the current node and keep traversing            
        self.node_point_ids[new_point_id] = new_point_id
        
        distance = VPTree.distance_by_id(self.vp_id, new_point_id)
        if distance >= (self.median - VPTree.tolerance):

            if distance > (self.right_max + VPTree.tolerance):
                self.right_max = distance

            if distance < (self.right_min - VPTree.tolerance):
                self.right_min = distance
                                     

            if self.right is None:
                self.right = VPTree({new_point_id:new_point_id}, self, new_point_id)
            else:
                self.right.__traverse(new_point_id)

        else:
            if distance > (self.left_max + VPTree.tolerance):
                self.left_max = distance

            if distance < (self.left_min - VPTree.tolerance):
                self.left_min = distance


            if self.left is None:
                self.left = VPTree({new_point_id:new_point_id}, self, new_point_id)
            else:
                self.left.__traverse(new_point_id)
        

    def remove_point(self, point_id : int):
        """ Removes the data point whose point id is 'point_id'
        
        Parameters
        ----------
        point_id : int
            Id of the point to be removed.
        """

        if point_id < 0 or point_id not in VPTree.points:
            raise ValueError('The point id must be a positive integer and be in the tree')
        
        self.__traverse_and_remove(point_id)

        del VPTree.points[point_id]
        del VPTree.vp_id_to_node[point_id]

    def __traverse_and_remove(self, point_id : int):
        """ Traverses the tree and removes the point with 'point_id'
        
        Parameters
        ----------
        point_id : int
            Point id to be removed from the VP-tree.
        """

        if self.vp_id == point_id:

            if self._is_leaf():
                return True #Flag that the point is in leaf

            # when the point is found, find new vp-point and reconstruct the subtree
            new_vp_id = -1
            new_vp_distance = 0.0            
            nodes_to_reconstruction = copy.deepcopy(self.node_point_ids)
            
            if self.parent == None:
                new_vp_id = next(iter(nodes_to_reconstruction))

            else:
                for p in iter(nodes_to_reconstruction.keys()):
                    dist = VPTree.distance_by_id(self.parent.vp_id, p)      
                    if dist > (new_vp_distance + VPTree.tolerance):
                        new_vp_distance = dist
                        new_vp_id = p

            self.__init__(nodes_to_reconstruction, self.parent, new_vp_id)     

        else: #if `point_id` is not the VP point of this node

            del self.node_point_ids[point_id]
            distance = VPTree.distance_by_id(self.vp_id, point_id)            

            if distance >= (self.median - VPTree.tolerance):

                if abs(distance - self.right_max) < VPTree.tolerance:
                    self.right_max = 0.0
                    for p in list(self.right.node_point_ids.keys()):
                        if p == point_id:
                            continue

                        dist = VPTree.distance_by_id(self.vp_id, p)
                        if dist > (self.right_max + VPTree.tolerance):
                            self.right_max = dist

                    if self.right.vp_id != point_id:
                        self.right_max = max(VPTree.distance_by_id(self.vp_id, self.right.vp_id), self.right_max)

                if abs(distance - self.right_min) < VPTree.tolerance:
                    self.right_min = np.inf                    
                    for p in list(self.right.node_point_ids.keys()):
                        if p == point_id:
                            continue

                        dist = VPTree.distance_by_id(self.vp_id, p)                    
                        if dist < (self.right_min - VPTree.tolerance):
                            self.right_min = dist
                    
                    if self.right.vp_id != point_id:
                        self.right_min = min(VPTree.distance_by_id(self.vp_id, self.right.vp_id), self.right_min)

                point_in_leaf = self.right.__traverse_and_remove(point_id)
                if point_in_leaf:
                    self.right = None
                    self.right_min = np.inf
                    self.right_max = 0.0

            else:
                if abs(distance - self.left_max) < VPTree.tolerance:                    
                    self.left_max = 0.0                
                    for p in list(self.left.node_point_ids.keys()):
                        if p == point_id:
                            continue

                        dist = VPTree.distance_by_id(self.vp_id, p)                    
                        if dist > (self.left_max + VPTree.tolerance):
                            self.left_max = dist
                    
                    if self.left.vp_id != point_id:
                        self.left_max = max(VPTree.distance_by_id(self.vp_id, self.left.vp_id), self.left_max)

                if abs(distance - self.left_min) < VPTree.tolerance:
                    self.left_min = np.inf
                    if self.left.vp_id != point_id:
                        self.left_min = VPTree.distance_by_id(self.vp_id, self.left.vp_id)

                    for p in list(self.left.node_point_ids.keys()):
                        if p == point_id:
                            continue

                        dist = VPTree.distance_by_id(self.vp_id, p)                    
                        if dist < (self.left_min - VPTree.tolerance):
                            self.left_min = dist
                    
                    if self.left.vp_id != point_id:
                        self.left_min = min(VPTree.distance_by_id(self.vp_id, self.left.vp_id), self.left_min)

                point_in_leaf = self.left.__traverse_and_remove(point_id)
                if point_in_leaf:
                    self.left = None
                    self.left_min = np.inf
                    self.left_max = 0.0
    
        return False


    def update_point(self, point_id : int, new_point):
        """ Update the point with 'point_id' to 'new_point'
        
        Parameters
        ----------
        point_id : int
            Point id to be removed from the VP-tree.
        new_point : List[float]
            The updated point.
        """
        # print("In tree:::: update before")
        # print("vp-tree:: ", list(VPTree.points.keys()), flush=True)
        self.remove_point(point_id)
        # print("In tree:::: update after")
        # print("vp-tree:: ", list(VPTree.points.keys()), flush=True)
        return self.add_point(new_point)
        

    def get_nearest_neighbor(self, query):
        """ Get single nearest neighbor.
        
        Parameters
        ----------
        query : List[float]
            Query point.

        Returns
        -------
        Tuple[float, List[float], int]
            Single nearest neighbor. (distance, point, vp_tree_id)
        """
        return self.get_n_nearest_neighbors(query, n_neighbors=1)[0]


    def get_n_nearest_neighbors_simple(self, query, n_neighbors):
        """ Get `n_neighbors` nearest neigbors to `query`
        
        Parameters
        ----------
        query : List[float]
            Query point.
        n_neighbors : int
            Number of neighbors to search.

        Returns
        -------
        List[Tuple[float, List[float]]]
            List of `n_neighbors` nearest neighbors: distances and points.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be strictly positive integer')
            
        neighbors = _AutoSortingList(max_size=n_neighbors)
        nodes_to_visit = [ self ]

        nth_distance = np.inf

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop()            
            
            if node is None:
                continue
                        
            dist = VPTree.dist_fn(query, VPTree.points[node.vp_id])
            
            if dist < (nth_distance - VPTree.tolerance):
                neighbors.append((dist, VPTree.points[node.vp_id], node.vp_id))
                
                # Update the furtherst distance after having found at least `n_neighbors`
                if len(neighbors) >= n_neighbors:
                    nth_distance = neighbors[-1][0]

            if node._is_leaf():
                continue

            if dist < node.median:
                if dist < (node.median + nth_distance):
                    nodes_to_visit.append(node.left)

                if dist >= (node.median - nth_distance):
                    nodes_to_visit.append(node.right)
            else:
                if dist >= (node.median - nth_distance):
                    nodes_to_visit.append(node.right)

                if dist < (node.median + nth_distance):
                    nodes_to_visit.append(node.left)
            
        return list(neighbors)

    def get_n_nearest_neighbors_original(self, query, n_neighbors):
        """ Get `n_neighbors` nearest neigbors to `query`.
        This is the search mechanism used by the data structure's seminal paper.
        
        Parameters
        ----------
        query : List[float]
            Query point.
        n_neighbors : int
            Number of neighbors to search.

        Returns
        -------
        List[Tuple[float, List[float], int]]
            List of `n_neighbors` nearest neighbors: distances, points, and vp-tree ids.
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be strictly positive integer')
            
        neighbors = _AutoSortingList(max_size=n_neighbors)
        nodes_to_visit = [ self ]

        nth_distance = np.inf

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop()            
            
            if node is None:
                continue
                        
            dist = VPTree.dist_fn(query, VPTree.points[node.vp_id])
            
            if dist < (nth_distance - VPTree.tolerance):
                neighbors.append((dist, VPTree.points[node.vp_id], node.vp_id))
                
                # Update the furtherst distance after having found at least `n_neighbors`
                if len(neighbors) >= n_neighbors:
                    nth_distance = neighbors[-1][0]

            if node._is_leaf():
                continue

            middle = (node.left_max + node.right_min) / 2.0

            if dist < middle:
                if node.left_min - nth_distance <= dist <= node.left_min + nth_distance:
                    nodes_to_visit.append(node.left)

                if node.right_min - nth_distance <= dist <= node.right_max + nth_distance:
                    nodes_to_visit.append(node.right)
            else:
                if node.right_min - nth_distance <= dist <= node.right_max + nth_distance:
                    nodes_to_visit.append(node.right)
                    
                if node.left_min - nth_distance <= dist <= node.left_min + nth_distance:
                    nodes_to_visit.append(node.left)                
            
        return list(neighbors)


    def get_n_nearest_neighbors(self, query, n_neighbors) -> List[Tuple[float, List[float], int]]:
        """ Get `n_neighbors` nearest neigbors to `query`
        
        Parameters
        ----------
        query : List[float]
            Query point.
        n_neighbors : int
            Number of neighbors to search.

        Returns
        -------
            List of `n_neighbors` nearest neighbors: distances, points, and vp-tree ids : List[Tuple[float, List[float], int]]
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
                neighbors.append((d, VPTree.points[node.vp_id], node.vp_id))
                
                # Update the furtherst distance after having found at least `n_neighbors`
                if len(neighbors) >= n_neighbors:
                    furthest_d, _, _ = neighbors[-1]

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
        query : List[float]
            Query point.
        max_distance : float
            Threshold distance for query.

        Returns
        -------
        neighbors : List[Tuple[float, List[float], int]]
            List of distances, points, and vp-tree ids within `max_distance`.

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
                neighbors.append((d, VPTree.points[node.vp_id], node.vp_id))                

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

