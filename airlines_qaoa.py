import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import dimod
import neal


class Airlines:
    """
    Generalized QAOA solver for airline routing optimization problem.
    """
    
    def __init__(self, afr_matrix, btr_matrix, route_costs=None, 
                 penalty_A=1.0, penalty_B=1.0, flight_penalties=None):
        """
        Initialize the QAOA problem.
        
        Parameters:
        -----------
        afr_matrix : numpy.ndarray (m x n)
            Flight-Route incidence matrix where m = number of flights, n = number of routes
            afr[f, r] = 1 if flight f is in route r, else 0
            
        btr_matrix : numpy.ndarray (k x n)
            Tail-Route incidence matrix where k = number of tails, n = number of routes
            btr[t, r] = 1 if tail t is assigned to route r, else 0
            
        route_costs : numpy.ndarray (n,), optional
            Cost of each route. Defaults to zeros.
            
        penalty_A : float
            Penalty coefficient for flight coverage constraints
            
        penalty_B : float
            Penalty coefficient for aircraft assignment constraints
            
        flight_penalties : numpy.ndarray (m,), optional
            Penalty for each uncovered flight (Cf for each flight f).
            Higher values mean more important flights.
            Defaults to ones (all flights equally important).
        """
        self.afr = np.array(afr_matrix)
        self.btr = np.array(btr_matrix)
        
        self.m_flights = self.afr.shape[0]  # Number of flights
        self.n_routes = self.afr.shape[1]   # Number of routes
        self.k_tails = self.btr.shape[0]    # Number of tails
        
        # Validate dimensions
        assert self.btr.shape[1] == self.n_routes, \
            "btr_matrix must have same number of columns as afr_matrix"
        
        # Set costs
        if route_costs is None:
            self.route_costs = np.zeros(self.n_routes)
        else:
            self.route_costs = np.array(route_costs)
            assert len(self.route_costs) == self.n_routes
        
        # Set flight penalties (Cf_f for each flight f)
        if flight_penalties is None:
            self.flight_penalties = np.ones(self.m_flights)
        else:
            self.flight_penalties = np.array(flight_penalties)
            assert len(self.flight_penalties) == self.m_flights, \
                f"flight_penalties must have length {self.m_flights} (number of flights)"
        
        self.A = penalty_A
        self.B = penalty_B
        
        # Compute QUBO coefficients
        self.J_matrix, self.h_vector = self._compute_qubo_coefficients()
        
        # Initialize quantum circuit parameters (will be set during circuit building)
        self.gamma_params = []
        self.beta_params = []
        self.circuit = None
        self.p_layers = None
        
    def _compute_qubo_coefficients(self):
        """
        Compute the coupling matrix J and linear coefficients h for the QUBO formulation.
        
        Returns:
        --------
        J_matrix : numpy.ndarray (n x n)
            Coupling coefficients where J[r, r'] represents interaction between routes r and r'
            
        h_vector : numpy.ndarray (n,)
            Linear coefficients for each route
        """
        n = self.n_routes
        J = np.zeros((n, n))
        
        # Compute coupling terms: J_{rr'} = 2A * sum_f(a_{fr} * a_{fr'}) + 2B * sum_t(b_{tr} * b_{tr'})
        for r in range(n):
            for r_prime in range(r + 1, n):
                # Flight constraint contribution
                flight_term = 2 * self.A * np.sum(self.afr[:, r] * self.afr[:, r_prime])
                
                # Tail constraint contribution
                tail_term = 2 * self.B * np.sum(self.btr[:, r] * self.btr[:, r_prime])
                
                J[r, r_prime] = flight_term + tail_term
                J[r_prime, r] = J[r, r_prime]  # Symmetric
        
        # Compute linear terms: h_r = c_r - sum_f(Cf_f * a_{fr}) - A * sum_f(a_{fr}) - B * sum_t(b_{tr})
        h = np.zeros(n)
        for r in range(n):
            # Route cost
            cost_term = self.route_costs[r]
            
            # Uncovered flight penalty (sum over flights, each with its own Cf_f)
            flight_penalty_term = np.sum(self.flight_penalties * self.afr[:, r])
            
            # Flight coverage constraint penalty
            flight_constraint_term = self.A * np.sum(self.afr[:, r])
            
            # Tail assignment constraint penalty
            tail_constraint_term = self.B * np.sum(self.btr[:, r])
            
            h[r] = cost_term - flight_penalty_term - flight_constraint_term - tail_constraint_term
        
        return J, h
    
    def build_qaoa(self, p_layers=1):
        """
        Build the QAOA quantum circuit.
        
        Parameters:
        -----------
        p_layers : int
            Number of QAOA layers (depth)
            
        Returns:
        --------
        circuit : QuantumCircuit
            The parameterized QAOA circuit
        """
        n = self.n_routes
        self.p_layers = p_layers
        qr = QuantumRegister(n, 'q')
        qc = QuantumCircuit(qr)
        
        # Create parameters for each layer
        self.gamma_params = [Parameter(f'γ_{k}') for k in range(p_layers)]
        self.beta_params = [Parameter(f'β_{k}') for k in range(p_layers)]
        
        # Initial state: uniform superposition |+>^n
        qc.h(range(n))
        qc.barrier()
        
        # Apply p layers of QAOA
        for layer in range(p_layers):
            # Cost Hamiltonian layer (Phase separation)
            self._apply_cost_unitary(qc, qr, self.gamma_params[layer])
            qc.barrier()
            
            # Mixer Hamiltonian layer
            self._apply_mixer_unitary(qc, qr, self.beta_params[layer])
            qc.barrier()
        
        # Measurement
        qc.measure_all()
        
        self.circuit = qc
        return qc
    
    def _apply_cost_unitary(self, qc, qr, gamma):
        """
        Apply the cost Hamiltonian unitary: exp(-i*gamma*H_C)
        
        H_C consists of:
        1. Quadratic terms: J_{rr'} * sigma_z_r * sigma_z_{r'}
        2. Linear terms: h_r * sigma_z_r
        """
        n = self.n_routes
        
        # Apply RZZ gates for quadratic coupling terms
        for r in range(n):
            for r_prime in range(r + 1, n):
                if self.J_matrix[r, r_prime] != 0:
                    # RZZ(2*theta) implements exp(-i*theta*Z_i*Z_j)
                    angle = 2 * gamma * self.J_matrix[r, r_prime]
                    qc.rzz(angle, qr[r], qr[r_prime])
        
        # Apply RZ gates for linear terms
        for r in range(n):
            if self.h_vector[r] != 0:
                angle = 2 * gamma * self.h_vector[r]
                qc.rz(angle, qr[r])
    
    def _apply_mixer_unitary(self, qc, qr, beta):
        """
        Apply the mixer Hamiltonian unitary: exp(-i*beta*H_M)
        
        H_M = -sum_i sigma_x_i
        """
        n = self.n_routes
        
        # Apply RX gates (RX(2*beta) implements exp(-i*beta*X))
        for r in range(n):
            qc.rx(2 * beta, qr[r])
    
    def compute_energy(self, bitstring):
        """
        Calculate the energy of a given bitstring solution.
        
        Parameters:
        -----------
        bitstring : str
            Binary string representing route selection (e.g., "0110")
            
        Returns:
        --------
        energy : float
            Total energy of the configuration
        """
        # Convert bitstring to array (reverse for correct indexing)
        x = np.array([int(bit) for bit in bitstring[::-1]])
        
        # Linear term
        linear_energy = np.dot(self.h_vector, x)
        
        # Quadratic term
        quadratic_energy = 0
        for r in range(self.n_routes):
            for r_prime in range(r + 1, self.n_routes):
                quadratic_energy += self.J_matrix[r, r_prime] * x[r] * x[r_prime]
        
        return linear_energy + quadratic_energy
    
    def compute_expectation(self, counts):
        """
        Compute the expectation value of the cost Hamiltonian.
        
        Parameters:
        -----------
        counts : dict
            Measurement results from quantum circuit
            
        Returns:
        --------
        expectation : float
            Average energy across all measured states
        """
        total_counts = sum(counts.values())
        expectation = 0
        
        for bitstring, count in counts.items():
            energy = self.compute_energy(bitstring)
            expectation += energy * (count / total_counts)
        
        return expectation
    
    def verify_solution(self, bitstring):
        """
        Verify if a bitstring satisfies all constraints.
        
        Parameters:
        -----------
        bitstring : str
            Binary string representing route selection
            
        Returns:
        --------
        valid : bool
            True if all constraints are satisfied
        details : dict
            Detailed constraint satisfaction information
        """
        x = np.array([int(bit) for bit in bitstring[::-1]])
        
        # Check flight coverage constraints
        flight_coverage = self.afr @ x
        flights_ok = np.all(flight_coverage == 1)
        
        # Check tail assignment constraints
        tail_assignment = self.btr @ x
        tails_ok = np.all(tail_assignment == 1)
        
        details = {
            'valid': flights_ok and tails_ok,
            'flights_covered': flight_coverage.tolist(),
            'flights_satisfied': flights_ok,
            'tails_assigned': tail_assignment.tolist(),
            'tails_satisfied': tails_ok,
            'selected_routes': [i for i, bit in enumerate(x) if bit == 1],
            'energy': self.compute_energy(bitstring)
        }
        
        return details['valid'], details
    
    def optimize(self, p_layers=1, initial_params=None, shots=1024, method='COBYLA'):
        """
        Run the complete QAOA optimization.
        
        Parameters:
        -----------
        p_layers : int
            Number of QAOA layers
        initial_params : list, optional
            Initial parameter values [γ_0, β_0, γ_1, β_1, ...] for p layers
            If None, uses [1.0, 1.0, ...] for all parameters
        shots : int
            Number of measurement shots
        method : str
            Classical optimization method
            
        Returns:
        --------
        result : dict
            Optimization results including optimal parameters and final counts
        """
        # Build circuit if not already built or if different p_layers
        if self.circuit is None or self.p_layers != p_layers:
            self.build_qaoa(p_layers)
        
        # Initialize simulator
        simulator = AerSimulator()
        
        # Objective function for classical optimizer
        def objective_function(params):
            # params = [γ_0, β_0, γ_1, β_1, ..., γ_{p-1}, β_{p-1}]
            # Create parameter binding dictionary
            param_dict = {}
            for k in range(p_layers):
                param_dict[self.gamma_params[k]] = params[2*k]
                param_dict[self.beta_params[k]] = params[2*k + 1]
            
            # Bind parameters
            bound_circuit = self.circuit.assign_parameters(param_dict)
            
            # Run simulation
            result = simulator.run(bound_circuit, shots=shots).result()
            counts = result.get_counts()
            
            return self.compute_expectation(counts)
        
        # Initial parameters: [γ_0, β_0, γ_1, β_1, ..., γ_{p-1}, β_{p-1}]
        if initial_params is None:
            initial_params = [1.0, 1.0] * p_layers
        
        assert len(initial_params) == 2 * p_layers, \
            f"Expected {2*p_layers} parameters for {p_layers} layers, got {len(initial_params)}"
        
        # Optimize
        print(f"Starting QAOA optimization with {p_layers} layer(s)...")
        print(f"Optimizing {2*p_layers} parameters: ", end="")
        for k in range(p_layers):
            print(f"(γ_{k}, β_{k})", end=" ")
        print()
        
        res = minimize(objective_function, initial_params, method=method)
        
        print(f"\nOptimization complete!")
        print(f"Optimal parameters:")
        for k in range(p_layers):
            print(f"  Layer {k}: γ_{k} = {res.x[2*k]:.4f}, β_{k} = {res.x[2*k+1]:.4f}")
        print(f"Final expectation: {res.fun:.4f}")
        
        # Run final circuit with optimal parameters
        param_dict = {}
        for k in range(p_layers):
            param_dict[self.gamma_params[k]] = res.x[2*k]
            param_dict[self.beta_params[k]] = res.x[2*k + 1]
        
        final_circuit = self.circuit.assign_parameters(param_dict)
        
        final_result = simulator.run(final_circuit, shots=5000).result()
        final_counts = final_result.get_counts()
        
        # Store optimal parameters in more readable format
        optimal_gammas = [res.x[2*k] for k in range(p_layers)]
        optimal_betas = [res.x[2*k+1] for k in range(p_layers)]
        
        return {
            'optimal_gammas': optimal_gammas,
            'optimal_betas': optimal_betas,
            'optimal_params': res.x,
            'expectation_value': res.fun,
            'counts': final_counts,
            'optimization_result': res,
            'p_layers': p_layers
        }
    
    def analyze_results(self, counts, top_k=5, valid_sol=True):
        """
        Analyze and display the top solutions.
        
        Parameters:
        -----------
        counts : dict
            Measurement counts
        top_k : int
            Number of top solutions to display
        valid_sol : bool
            Prints valid solutions if True, otherwise prints all solutions with their validity status
            
        Returns:
        --------
        solutions : list
            List of top solutions with their details
        """
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        solutions = []
        if valid_sol:
            print(f"\n{'='*80}")
            print(f"Valid Solution Found:")
            print(f"{'='*80}")
            for i, (bitstring, count) in enumerate(sorted_counts):
                valid, details = self.verify_solution(bitstring)
                if valid:
                    print(f"\n{i+1}. Bitstring: {bitstring}  Counts: {count}  Energy: {details['energy']:.4f}")
            
        print(f"\n{'='*80}")
        print(f"Top {top_k} Solutions:")
        print(f"{'='*80}")
        
        for i, (bitstring, count) in enumerate(sorted_counts[:top_k]):
            valid, details = self.verify_solution(bitstring)
            energy = self.compute_energy(bitstring)
            
            print(f"\n{i+1}. Bitstring: {bitstring}")
            print(f"   Counts: {count}")
            print(f"   Energy: {energy:.4f}")
            print(f"   Valid: {'✓' if valid else '✗'}")
            print(f"   Selected Routes: {details['selected_routes']}")
            
            if not valid:
                print(f"   Flight Coverage: {details['flights_covered']} (should be all 1s)")
                print(f"   Tail Assignment: {details['tails_assigned']} (should be all 1s)")
            
            solutions.append({
                'bitstring': bitstring,
                'count': count,
                'energy': energy,
                'valid': valid,
                'details': details
            })
        
        return solutions
    
    def problem_summary(self):
        """Print a summary of the problem setup."""
        print(f"\n{'='*80}")
        print(f"Airline Routing Problem Summary")
        print(f"{'='*80}")
        print(f"Number of Routes: {self.n_routes}")
        print(f"Number of Flights: {self.m_flights}")
        print(f"Number of Tails: {self.k_tails}")
        print(f"\nPenalty Coefficients:")
        print(f"  A (Flight Coverage Constraint): {self.A}")
        print(f"  B (Tail Assignment Constraint): {self.B}")
        print(f"\nFlight Penalties (Cf for each flight):")
        for f in range(self.m_flights):
            print(f"  Flight {f+1}: Cf_{f+1} = {self.flight_penalties[f]}")
        print(f"\nRoute Costs:")
        for r in range(self.n_routes):
            print(f"  Route {r+1}: c_{r+1} = {self.route_costs[r]}")
        print(f"\nFlight-Route Incidence Matrix (afr):")
        print(self.afr)
        print(f"\nTail-Route Incidence Matrix (btr):")
        print(self.btr)
        print(f"\nQUBO Linear Coefficients (h):")
        print(self.h_vector)
        print(f"\nQUBO Coupling Matrix (J):")
        print(self.J_matrix)
        print(f"{'='*80}\n")

    def simulated_annealing(self, num_reads=100, initial_state=None, beta_range=None, 
                        beta_schedule_type='geometric', seed=None):
        """
        Solve the problem using classical simulated annealing (for comparison).
        
        Parameters:
        -----------
        num_reads : int
            Number of reads/samples to generate (default: 100)
        initial_state : dict, optional
            Initial state for the annealing process. If None, uses random initial state.
        beta_range : list or tuple, optional
            A 2-tuple defining the beginning and end of the beta schedule.
            Beta is the inverse temperature. If None, uses sampler's default.
        beta_schedule_type : str
            Type of beta schedule: 'geometric' or 'linear' (default: 'geometric')
        seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        result : dict
            Contains the best solution found and its energy
        """
        # Create a Binary Quadratic Model (BQM) for dimod
        h = {i: self.h_vector[i] for i in range(self.n_routes)}
        J = {(i, j): self.J_matrix[i, j] 
            for i in range(self.n_routes) 
            for j in range(i + 1, self.n_routes) 
            if self.J_matrix[i, j] != 0}
        
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
        
        # Use Neal's Simulated Annealing Sampler
        sampler = neal.SimulatedAnnealingSampler()
        
        # Build kwargs dictionary with only non-None values
        sample_kwargs = {'num_reads': num_reads}
        
        if initial_state is not None:
            sample_kwargs['initial_state'] = initial_state
        if beta_range is not None:
            sample_kwargs['beta_range'] = beta_range
        if beta_schedule_type is not None:
            sample_kwargs['beta_schedule_type'] = beta_schedule_type
        if seed is not None:
            sample_kwargs['seed'] = seed
        
        # Run simulated annealing
        sampleset = sampler.sample(bqm, **sample_kwargs)
        
        # Get the best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Convert sample dict to bitstring for consistency with QAOA results
        bitstring = ''.join(str(best_sample[i]) for i in range(self.n_routes))
        
        print(f"\nSimulated Annealing Result:")
        print(f"Best Bitstring: {bitstring}")
        print(f"Best Energy: {best_energy:.4f}")
        
        # Verify the solution
        valid, details = self.verify_solution(bitstring)
        print(f"Valid Solution: {'✓' if valid else '✗'}")
        
        return {
            'best_sample': best_sample,
            'best_bitstring': bitstring,
            'best_energy': best_energy,
            'valid': valid,
            'details': details,
            'sampleset': sampleset
        }

    def plot_sa_histogram(self, sa_result, top_k=10, figsize=(12, 6)):
        """
        Plot histogram of simulated annealing results.
        
        Parameters:
        -----------
        sa_result : dict
            Result dictionary from simulated_annealing method
        top_k : int
            Number of top solutions to display in the histogram
        figsize : tuple
            Figure size (width, height)
        """
        sampleset = sa_result['sampleset']
        
        # Convert sampleset to counts dictionary (bitstring -> count)
        counts = {}
        for sample, energy, num_occurrences in sampleset.data(['sample', 'energy', 'num_occurrences']):
            # Convert sample dict to bitstring
            bitstring = ''.join(str(sample[i]) for i in range(self.n_routes))[::-1]
            if bitstring in counts:
                counts[bitstring] += num_occurrences
            else:
                counts[bitstring] = num_occurrences
        
        # Sort by counts and get top_k
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Prepare data for plotting
        bitstrings = [item[0] for item in sorted_counts]
        frequencies = [item[1] for item in sorted_counts]
        
        # Create color coding based on validity
        colors = []
        for bitstring in bitstrings:
            valid, _ = self.verify_solution(bitstring)
            colors.append('green' if valid else 'red')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(bitstrings)), frequencies, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize the plot
        ax.set_xlabel('Solution (Bitstring)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Simulated Annealing Results - Top Solutions', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bitstrings)))
        ax.set_xticklabels(bitstrings, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{freq}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Valid Solution'),
            Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Invalid Solution')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Simulated Annealing Histogram Summary")
        print(f"{'='*80}")
        print(f"Total unique solutions found: {len(counts)}")
        print(f"Displaying top {len(bitstrings)} solutions")
        
        valid_count = sum(1 for bs in bitstrings if self.verify_solution(bs)[0])
        print(f"Valid solutions in top {top_k}: {valid_count}/{len(bitstrings)}")
        
        return fig, ax

    def plot_qaoa_histogram(self, qaoa_result, top_k=10, figsize=(12, 6)):
        """
        Plot histogram of QAOA results.
        
        Parameters:
        -----------
        qaoa_result : dict
            Result dictionary from run_qaoa method
        top_k : int
            Number of top solutions to display in the histogram
        figsize : tuple
            Figure size (width, height)
        """
        counts = qaoa_result['counts']
        
        # Sort by counts and get top_k
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Prepare data for plotting
        bitstrings = [item[0] for item in sorted_counts]
        frequencies = [item[1] for item in sorted_counts]
        
        # Create color coding based on validity
        colors = []
        energies = []
        for bitstring in bitstrings:
            valid, _ = self.verify_solution(bitstring)
            colors.append('green' if valid else 'red')
            energies.append(self.compute_energy(bitstring))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(bitstrings)), frequencies, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize the plot
        ax.set_xlabel('Solution (Bitstring)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'QAOA Results - Top Solutions (p={qaoa_result["p_layers"]} layers)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bitstrings)))
        ax.set_xticklabels(bitstrings, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{freq}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Valid Solution'),
            Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Invalid Solution')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"QAOA Histogram Summary")
        print(f"{'='*80}")
        print(f"Total unique solutions found: {len(counts)}")
        print(f"Displaying top {len(bitstrings)} solutions")
        print(f"QAOA layers (p): {qaoa_result['p_layers']}")
        
        valid_count = sum(1 for bs in bitstrings if self.verify_solution(bs)[0])
        print(f"Valid solutions in top {top_k}: {valid_count}/{len(bitstrings)}")
        
        # Show best valid solution
        for i, bs in enumerate(bitstrings):
            valid, details = self.verify_solution(bs)
            if valid:
                print(f"\nBest valid solution:")
                print(f"  Bitstring: {bs}")
                print(f"  Frequency: {frequencies[i]}")
                print(f"  Energy: {energies[i]:.4f}")
                print(f"  Selected routes: {details['selected_routes']}")
                break
        
        return fig, ax
