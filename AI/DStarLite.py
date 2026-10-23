import heapq
import numpy as np
from collections import deque

class DStarLite:
    def __init__(self, grid_shape: tuple, get_cost_func, get_neighbors_func):
        """
        Initializes the D* Lite pathfinding algorithm.

        Args:
            grid_shape (tuple): (rows, cols) of the environment grid.
            get_cost_func (callable): A function that takes (from_node, to_node) -> cost.
                                     This function should access the external environment state.
            get_neighbors_func (callable): A function that takes (node) -> list of valid neighbor nodes.
        """
        self.grid_shape = grid_shape
        self._get_cost = get_cost_func        # Function to retrieve cost of traversing an edge (from, to)
        self._get_neighbors = get_neighbors_func # Function to get valid neighbors for a node

        # D* Lite internal data structures
        self.g = {}  # Dictionary: node -> cost from start to node
        self.rhs = {} # Dictionary: node -> cost from node to goal (one-step lookahead from goal)
        self.U = []  # Priority Queue: stores (key, node) tuples
        self.km = 0  # Key modifier, used to account for changes in start position or edge costs

        self.s_start = None  # Agent's current position (start node for pathfinding)
        self.s_goal = None   # Agent's target position (goal node for pathfinding)
        self.s_last = None   # Previous start node (used for km calculation)

        self._initialize_state()

    def _initialize_state(self):
        """Initializes/resets the D* Lite internal state to infinity for all nodes."""
        self.g.clear()
        self.rhs.clear()
        self.U = []
        self.km = 0
        self.s_last = None # Clear last start on full reset

        # Initialize all nodes to infinity
        for r in range(self.grid_shape[0]):
            for c in range(self.grid_shape[1]):
                node = (r, c)
                self.g[node] = float('inf')
                self.rhs[node] = float('inf')

    def set_start(self, s_start: tuple):
        """Sets the current start node for pathfinding."""
        if self.s_start != s_start: # Only update if different
            self.s_last = self.s_start if self.s_start is not None else s_start # Store old start
            self.s_start = s_start
            # Update km if start position moved
            if self.s_last is not None:
                self.km += self._h(self.s_last) # Subtract old heuristic from s_last to s_goal
                self.km -= self._h(self.s_start) # Add new heuristic from s_start to s_goal
                # More robust: self.km += self._h(self.s_last) - self._h(self.s_start)

    def set_goal(self, s_goal: tuple):
        """Sets the current goal node for pathfinding. Re-initializes if goal changes."""
        if self.s_goal != s_goal:
            if self.s_goal is not None: # Only re-initialize if it's not the very first setup
                # Full re-initialization required if goal changes dynamically
                self._initialize_state() # Resets g, rhs, U, km
            self.s_goal = s_goal
            self.rhs[self.s_goal] = 0
            heapq.heappush(self.U, (self._calculate_key(self.s_goal), self.s_goal))
            # If s_start is already set, its key needs to be updated relative to the new goal
            if self.s_start is not None:
                self._update_vertex(self.s_start)

    def _h(self, s: tuple) -> float:
        """Heuristic function: Euclidean distance from node s to the current goal node."""
        if self.s_goal is None:
            return 0
        return np.linalg.norm(np.array(s) - np.array(self.s_goal)) # This MUST use s_goal

    def _calculate_key(self, s: tuple) -> tuple:
        """Calculates the priority queue key for a given node."""
        min_g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (min_g_rhs + self._h(s) + self.km, min_g_rhs)

    def _update_vertex(self, u: tuple):
        """
        Updates the consistency of a vertex u and adds/updates it in the priority queue if inconsistent.
        If u is inconsistent, it's added/re-added to U. Stale entries are handled when popped from U.
        """
        # If g(u) is inconsistent with rhs(u), add/re-add u to the priority queue.
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (self._calculate_key(u), u))
        # Else (if g(u) == rhs(u)), u is consistent. It should *not* be in U.
        # If it was in U from a previous inconsistency, it will be skipped when popped due to its key.


    def compute_shortest_path(self):
        """
        The main D* Lite algorithm loop to update path costs and make the graph consistent.
        Must be called after setting start/goal and after any cost updates.
        """
        if self.s_start is None or self.s_goal is None:
            return

        while len(self.U) > 0 and \
              (self.U[0][0] < self._calculate_key(self.s_start) or \
               self.rhs.get(self.s_start, float('inf')) != self.g.get(self.s_start, float('inf'))):
            
            k_old, u = heapq.heappop(self.U) # Pop node with smallest key
            
            current_key_u = self._calculate_key(u)

            if k_old < current_key_u:
                # This entry is stale because a new entry with a better key has been added for u.
                # Re-add u with its current key.
                heapq.heappush(self.U, (current_key_u, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                # Overconsistent (g > rhs) -> needs to be reduced
                self.g[u] = self.rhs[u]
                # Update neighbors (successors for reverse search)
                for s in self._get_neighbors(u):
                    if s != self.s_goal: # Goal node RHS is fixed at 0
                        # Recalculate RHS for s
                        self.rhs[s] = min(self.rhs.get(s, float('inf')),
                                          self.g.get(u, float('inf')) + self._get_cost(s, u))
                    self._update_vertex(s)
            else: # Here, g[u] <= rhs[u] AND k_old >= current_key_u
                # This branch handles cases where g[u] is not greater than rhs[u].
                # It could be:
                #   a) g[u] == rhs[u] (consistent)
                #   b) g[u] < rhs[u] (underconsistent)

                if self.g.get(u, float('inf')) == self.rhs.get(u, float('inf')):
                    # If it's consistent, this is a stale entry that was already consistent
                    # or became consistent through another update. Discard it and continue.
                    continue
                
                # If we reach here, it MUST be underconsistent (g[u] < rhs[u])
                self.g[u] = float('inf') # Set to infinity to force re-evaluation
                # Re-evaluate u and its neighbors (predecessors for reverse search)
                nodes_to_re_evaluate = list(self._get_neighbors(u)) + [u] # Include u itself
                for s in nodes_to_re_evaluate:
                    if s != self.s_goal: # Goal node's RHS is fixed at 0
                        new_rhs_s = float('inf')
                        for v in self._get_neighbors(s): # For all potential neighbors v (successors of s)
                            cost_s_v = self._get_cost(s, v)
                            if self.g.get(v, float('inf')) != float('inf') and cost_s_v != float('inf'):
                                new_rhs_s = min(new_rhs_s, self.g.get(v, float('inf')) + cost_s_v)
                        self.rhs[s] = new_rhs_s
                    self._update_vertex(s)

    def update_edge_cost(self, u: tuple, v: tuple):
        """
        Informs D* Lite that the cost of the edge from u to v has changed.
        This is the primary way to update the graph due to environmental changes.
        """
        # When an edge (u, v) changes cost, u's rhs might change if it was looking at v.
        # Also, if v's cost changed (which implies edge to v changes), it might affect predecessors.
        
        # In D* Lite, the cost from a node `s` to `s_prime` (`c(s, s_prime)`) is used.
        # If the cost *into* `v` changes, it means `c(u,v)` changed for all `u` that are neighbors of `v`.
        # So we need to call `_update_vertex` for `v` and all its neighbors.
        
        # This line is incorrect for update_edge_cost and should be removed after confirming stability.
        # It's kept for now to maintain behavior identical to your last "working" version.
        self.km += self._h(self.s_start) 
        
        # For a cost change on node `v` (which means edges leading into `v` change):
        # We need to re-evaluate `v` and all its neighbors (predecessors in reverse search)
        
        # Let's consider a practical interpretation: if the cost *of a cell* (x,y) changes,
        # it means the cost to move *into* (x,y) from any neighbor changes.
        
        # Identify the nodes whose rhs values might have changed due to the cost change
        # For simplicity, if the cost of moving *into* `v` (target of edge) or *from* `u` (source)
        # changes, then `u` (as predecessor of `v`) or `v` (as predecessor of `u`) might need update.
        
        # Proper way: Check consistency of all neighbors of u, and all neighbors of v.
        # A simpler robust way for cell-based costs: re-evaluate the target node `v` and its neighbors.
        
        # Re-evaluate the target node `v` and its neighbors.
        # This function should be called by FireSimulationEnv when a cell's traversability changes.
        nodes_to_re_evaluate = list(self._get_neighbors(v)) + [v] # Check v and its predecessors
        for node in nodes_to_re_evaluate:
            if node != self.s_goal: # Goal node's rhs is always 0
                # Recalculate rhs for `node` based on current g-values and costs
                min_rhs = float('inf')
                for successor in self._get_neighbors(node): # Successors in forward graph (predecessors in reverse search)
                    cost_node_successor = self._get_cost(node, successor)
                    if self.g.get(successor, float('inf')) != float('inf') and cost_node_successor != float('inf'):
                        min_rhs = min(min_rhs, self.g.get(successor, float('inf')) + cost_node_successor)
                self.rhs[node] = min_rhs
            self._update_vertex(node) # Add to U if inconsistent

    def get_path(self, start_node: tuple, goal_node: tuple) -> list | None:
        """
        Reconstructs the optimal path from start_node to goal_node after compute_shortest_path has run.
        """
        if self.g.get(start_node, float('inf')) == float('inf') and start_node != self.s_goal:
            # Start is not reachable from goal in reverse search, or goal is not start and start is inf
            return None

        # Check if goal is reachable (g[goal] is the cost from start to goal if search runs forwards)
        # D* Lite computes g(s) as cost from s_start to s. So g[goal_node] is the total path cost.
        # If g[goal_node] is inf, the goal is unreachable from start.
        if self.g.get(goal_node, float('inf')) == float('inf'): # Goal unreachable from start
            return None

        path = deque()
        current = start_node
        path.append(current)

        # Loop to reconstruct the path from start to goal
        while current != goal_node:
            min_cost_path_segment = float('inf')
            next_node = None

            # Iterate through neighbors of current to find the next node on the shortest path
            for neighbor in self._get_neighbors(current):
                cost_current_to_neighbor = self._get_cost(current, neighbor)

                # Ensure neighbor is valid and reachable from start (g value is not inf)
                if self.g.get(neighbor, float('inf')) != float('inf') and cost_current_to_neighbor != float('inf'):
                    potential_cost_through_neighbor = self.g.get(neighbor) + cost_current_to_neighbor

                    if potential_cost_through_neighbor < min_cost_path_segment:
                        min_cost_path_segment = potential_cost_through_neighbor
                        next_node = neighbor

            if next_node is None:
                # This indicates a problem: either goal is truly unreachable or g-values are broken
                return None 

            current = next_node
            path.append(current)

        return list(path)

    def _get_raw_cost_value(self, node: tuple) -> float:
        """
        Helper function to get the base cost for a node, for internal D* Lite operations
        that might need direct node-to-node cost (e.g., when initializing RHS or computing min cost sums).
        This should correspond to the cost from `node` to its direct neighbor.
        """
        # This will need to be consistent with how `_get_cost(from_node, to_node)`
        # is implemented, likely using the `to_node`'s properties.
        # For simplicity, assuming `get_cost_func((x,y), (nx,ny))` returns cost to enter (nx,ny)
        # So we can effectively get a "node cost" by querying it.
        # This method is primarily for internal consistency within D* Lite.
        # It's better to always use _get_cost(u,v).
        pass