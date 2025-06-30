# %%
import cplex
import time
import networkx as nx
from cplex.exceptions import CplexSolverError


def read_dimacs(path):
    """Parse DIMACS graph coloring instance"""
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("c"):
                continue  # Skip comments
            elif line.startswith("p"):
                _, _, nodes, edges = line.split()
                G.add_nodes_from(range(int(nodes)))
            elif line.startswith("e"):
                _, u, v = line.split()
                G.add_edge(int(u) - 1, int(v) - 1)  # Convert to 0-indexed
    return G


def find_maximal_cliques(graph):
    """Uses greedy algorithm to find at least one maximal clique for
    every node in a graph.
    """
    cliques = []
    remaining_nodes = set(graph.nodes())

    while remaining_nodes:
        node = list(remaining_nodes)[0]
        clique = set([node])
        neighbors = set(graph.neighbors(node))

        # Greedy search of cliques
        while neighbors:
            # Gets node with maximum number of neighbors in common
            # & acts as intersection of sets
            node = max(
                neighbors, key=lambda x: len(neighbors & set(graph.neighbors(x)))
            )
            clique.add(node)
            neighbors &= set(graph.neighbors(node))

        cliques.append(list(clique))
        remaining_nodes -= clique

    print(f"Found {len(cliques)} maximal cliques with greedy algorithm")
    return cliques


class ColumnGeneration:
    def __init__(self, graph, single_thread=False):
        self.n = graph.number_of_nodes()
        self.graph = graph
        self.complement_graph = nx.complement(self.graph)
        self.maximal_cliques = list(nx.find_cliques(self.complement_graph))
        self.initial_cliques = list(find_maximal_cliques(self.complement_graph))
        self.restricted_master = cplex.Cplex()
        self.dual = cplex.Cplex()
        self._create_rmp()
        self._create_dual()
        # Disable multithreading
        if single_thread:
            self.restricted_master.parameters.threads.set(1)
            self.dual.parameters.threads.set(1)

    def _create_rmp(self):
        self.restricted_master.set_log_stream(None)
        self.restricted_master.set_results_stream(None)
        self.restricted_master.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[], val=[]) for _ in range(self.n)],
            senses=["G"] * self.n,  # Sum >= 1
            rhs=[1.0] * self.n,
            names=[f"v{i}" for i in range(self.n)],
        )

        # Add initial columns from maximal cliques
        for clique in self.initial_cliques:
            self._add_column_to_rmp(clique)
        # for i in range(self.n):
        #     self._add_column_to_rmp([i])
        # covered = set()
        # sorted_cliques = sorted(self.maximal_cliques, key=len, reverse=True)
        # for clique in sorted_cliques:
        #     if not set(clique).issubset(covered):
        #         self._add_column_to_rmp(clique)
        #         covered.update(clique)
        #     if len(covered) == self.n:
        #         break

    def _add_column_to_rmp(self, clique):
        """Add new column for a clique in complement graph (Independent
        set in original graph)
        """
        # Create coefficient vector: 1 for vertices in clique, 0 otherwise
        coeffs = [1.0 if i in clique else 0.0 for i in range(self.n)]

        # Sparse representation (only non-zero coefficients)
        indices = [i for i, c in enumerate(coeffs) if c > 0]
        values = [c for c in coeffs if c > 0]

        # Add to RMP
        self.restricted_master.variables.add(
            obj=[1.0],  # Each clique costs 1 in objective
            columns=[cplex.SparsePair(ind=indices, val=values)],
            # LP relaxation
            types=[self.restricted_master.variables.type.continuous],
            names=[f"clique_{len(self.restricted_master.variables.get_names())}"],
        )
        self.restricted_master.set_problem_type(cplex.Cplex.problem_type.LP)

    def _create_dual(self):
        """Initialize Pricing Problem (Max Weight Clique in complement
        graph)
        """
        self.dual.set_log_stream(None)
        self.dual.set_results_stream(None)
        # Binary variables for vertices
        self.dual.variables.add(
            names=[f"y{i}" for i in range(self.n)],
            types=["B"] * self.n,  # Binary variables
            lb=[0] * self.n,
            ub=[1] * self.n,
        )

        constraints = []
        for u, v in self.graph.edges():
            # In original graph: if edge exists, cannot have both vertices
            # This enforces independent set in original = clique in complement
            constraint = cplex.SparsePair(ind=[f"y{u}", f"y{v}"], val=[1.0, 1.0])
            constraints.append(constraint)

        self.dual.linear_constraints.add(
            lin_expr=constraints,
            senses=["L"] * len(constraints),  # y_u + y_v <= 1
            rhs=[1.0] * len(constraints),
        )

        # Maximization objective (weights set later)
        self.dual.objective.set_sense(self.dual.objective.sense.maximize)

    def solve(self, tol=1e-6, max_iter=100, time_limit=300):
        """Solve linear relaxation via column generation
        Args:
            tol: Tolerance for reduced cost check
            max_iter: Maximum number of column generation iterations
            time_limit: Time limit in seconds

        Returns:
            dict: Results including fractional chi, times, and iterations
        """
        start_time = time.time()
        total_rmp_time = 0
        total_pricing_time = 0
        iterations = 0
        columns_added = 0

        # Set time limits
        self.restricted_master.parameters.timelimit.set(time_limit)
        self.dual.parameters.timelimit.set(time_limit)

        # For some obscure reason, cplex occasionally fails when not using multithreading
        # This avoids cplex error in these cases:
        last_feasible_obj = None

        # Main column generation loop
        for iteration in range(max_iter):
            # Solve RMP
            rmp_start = time.time()
            if (
                self.restricted_master.get_problem_type()
                != self.restricted_master.problem_type.LP
            ):
                print(f"Warning: RMP became MILP at iteration {iteration}")
                self.restricted_master.set_problem_type(
                    self.restricted_master.problem_type.LP
                )
            # covered_by = [0] * problem.n
            # for name in problem.restricted_master.variables.get_names():
            #     col = problem.restricted_master.variables.get_cols(name)
            #     for i in col.ind:
            #         covered_by[i] += 1
            #
            # uncovered = [i for i, count in enumerate(covered_by) if count == 0]
            try:
                self.restricted_master.solve()
            except CplexSolverError as e:
                print(f"RMP failed at itreation {iteration}: {e}")
            # print("Initial RMP objective value:", problem.restricted_master.solution.get_objective_value())

            # print("Is MIP:", self.restricted_master.problem_type[self.restricted_master.get_problem_type()])
            rmp_time = time.time() - rmp_start
            total_rmp_time += rmp_time
            rmp_status = self.restricted_master.solution.get_status()

            # Check RMP solution status
            if rmp_status != self.restricted_master.solution.status.optimal:
                print(f"RMP failed at iteration {iteration}")
                break
            else:
                last_feasible_obj = (
                    self.restricted_master.solution.get_objective_value()
                )

            # Get dual values for vertex constraints
            duals = self.restricted_master.solution.get_dual_values(
                [f"v{i}" for i in range(self.n)]
            )

            # Update pricing objective with duals as weights
            self.dual.objective.set_linear([(f"y{i}", duals[i]) for i in range(self.n)])

            # Solve pricing problem
            pricing_start = time.time()
            self.dual.solve()
            pricing_time = time.time() - pricing_start
            total_pricing_time += pricing_time
            # Check pricing solution status
            if self.dual.solution.get_status() != self.dual.solution.status.MIP_optimal:
                print(f"Pricing failed at iteration {iteration}")
                break

            # Get solution and calculate reduced cost
            max_weight = self.dual.solution.get_objective_value()
            reduced_cost = 1 - max_weight

            # Check termination condition
            if reduced_cost > -tol:  # No negative reduced cost
                print(
                    f"Terminated at iteration {iteration}: reduced cost = {reduced_cost:.6f}"
                )
                break

            # Get new clique (independent set in original graph)
            y_vals = self.dual.solution.get_values()
            new_clique = [i for i, val in enumerate(y_vals) if val > 0.5]

            if not new_clique:
                print(
                    f"Iteration {iteration}: Warning — empty clique returned by pricing."
                )
                break

            # Add new column to RMP
            self._add_column_to_rmp(new_clique)
            iterations += 1
            columns_added += 1
            if iterations >= max_iter:
                # Solve one last time after adding the last column
                self.restricted_master.solve()
            # print(f"Iteration {iteration}: Added clique (size={len(new_clique)}) with reduced cost {reduced_cost:.4f}")

        total_time = time.time() - start_time
        try:
            fractional_chi = self.restricted_master.solution.get_objective_value()
        except CplexSolverError:
            fractional_chi = last_feasible_obj
        init_msg = f"Graph has {len(self.graph.nodes)} nodes and "
        init_msg += f"{len(self.graph.edges)} edges"
        print(
            (
                f"{init_msg}\n{'-'*len(init_msg)}\n"
                f"Chromatic number (Chi): {fractional_chi}\n"
                f"Total time: {total_time}\nRMP time: {total_rmp_time}\n"
                f"Pricing time: {total_pricing_time}\n"
                f"Iterations: {iterations}\n"
                f"Number of initial columns: {len(self.initial_cliques)}"
            )
        )

        return {
            "fractional_chi": fractional_chi,
            "total_time": total_time,
            "rmp_time": total_rmp_time,
            "pricing_time": total_pricing_time,
            "iterations": iterations,
        }

for i in range(3):
    print(f"\n#Solving problem gcol{i+1}")
    g1 = read_dimacs(f"./data/gcol{i+1}.txt")
    problem = ColumnGeneration(g1, single_thread=False)
    res = problem.solve(max_iter=1000, tol=1e-6, time_limit=3000)
# test = nx.complement(g1)
# g1 = nx.mycielski_graph(5)
# cliques = list(nx.find_cliques(test))
# print(len(cliques))
# cliques[4]
# print(f"""Chi: {res['fractional_chi']}\nTotal time: {res['total_time']}
# RMP Time: {res['rmp_time']}\nPricing Time: {res['pricing_time']}
# Iterations: {res['iterations']}
# """)
# %%
problem.restricted_master.solve()
problem.restricted_master.solution.get_dual_values()
problem.dual.solve()
print(cplex.Cplex().get_version())
print(cplex.__file__)
problem.restricted_master.write("debug.lp")
names = problem.restricted_master.variables.get_names()
for name in names[:5]:
    col = problem.restricted_master.variables.get_cols(name)
    print(name, col)


print("Non-zero entries per constraint:")
for i in range(problem.n):
    row = problem.restricted_master.linear_constraints.get_rows(i)
    print(f"v{i}: {len(row.ind)} non-zeros")

len(g1.nodes())
