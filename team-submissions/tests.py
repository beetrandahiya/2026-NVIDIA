"""
LABS Problem - Comprehensive Test Suite
========================================
Milestone 3, Step A: CPU Validation

This test suite validates:
1. Energy function correctness
2. LABS symmetry properties  
3. MTS algorithm correctness
4. Interaction indices (G2, G4)
5. Quantum kernel properties (when CUDA-Q available)

Run with: pytest tests.py -v
Or:       python tests.py (standalone mode)

Team: Eigencrew
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Optional
from itertools import product

# ============================================================================
# LABS Core Functions (copied from notebook for standalone testing)
# ============================================================================

def compute_energy(sequence: np.ndarray) -> int:
    """
    Compute the LABS energy E(s) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=0}^{N-k-1} s_i * s_{i+k}
    """
    N = len(sequence)
    E = 0
    for k in range(1, N):
        C_k = np.dot(sequence[:N-k], sequence[k:])
        E += C_k * C_k
    return int(E)


def compute_autocorrelation(sequence: np.ndarray) -> List[int]:
    """Compute all autocorrelation values C_k for k=1 to N-1"""
    N = len(sequence)
    return [int(np.dot(sequence[:N-k], sequence[k:])) for k in range(1, N)]


def get_interactions(N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate interaction indices for the LABS Hamiltonian.
    G2: 2-body terms (pairs [i, j])
    G4: 4-body terms (quadruples [i, j, k, l])
    """
    G2 = []
    G4 = []
    
    for k in range(1, N):
        for i in range(N - k):
            j = i + k
            G2.append([i, j])
            
            for kp in range(k + 1, N):
                for ip in range(N - kp):
                    jp = ip + kp
                    if len(set([i, j, ip, jp])) == 4:
                        quad = sorted([i, j, ip, jp])
                        if quad not in G4:
                            G4.append(quad)
    
    return G2, G4


def brute_force_optimal(N: int) -> Tuple[int, np.ndarray]:
    """Find optimal LABS sequence by exhaustive search (only for small N)"""
    best_energy = float('inf')
    best_sequence = None
    
    for bits in product([1, -1], repeat=N):
        seq = np.array(bits)
        energy = compute_energy(seq)
        if energy < best_energy:
            best_energy = energy
            best_sequence = seq.copy()
    
    return int(best_energy), best_sequence


def initialize_population_random(N: int, pop_size: int) -> List[np.ndarray]:
    """Initialize a random population of binary sequences"""
    return [np.random.choice([1, -1], size=N) for _ in range(pop_size)]


def tabu_search(sequence: np.ndarray, max_iter: int = 100, 
                tabu_tenure: int = 7) -> Tuple[np.ndarray, int]:
    """
    Simple tabu search for LABS optimization.
    Returns: (best_sequence, best_energy)
    """
    N = len(sequence)
    current = sequence.copy()
    current_energy = compute_energy(current)
    
    best = current.copy()
    best_energy = current_energy
    
    tabu_list = {}
    
    for iteration in range(max_iter):
        best_neighbor = None
        best_neighbor_energy = float('inf')
        best_flip = -1
        
        for i in range(N):
            if tabu_list.get(i, 0) > iteration:
                if current_energy <= best_energy:
                    pass  # Aspiration criterion
                else:
                    continue
            
            neighbor = current.copy()
            neighbor[i] *= -1
            neighbor_energy = compute_energy(neighbor)
            
            if neighbor_energy < best_neighbor_energy:
                best_neighbor = neighbor
                best_neighbor_energy = neighbor_energy
                best_flip = i
        
        if best_neighbor is None:
            break
            
        current = best_neighbor
        current_energy = best_neighbor_energy
        tabu_list[best_flip] = iteration + tabu_tenure
        
        if current_energy < best_energy:
            best = current.copy()
            best_energy = current_energy
    
    return best, best_energy


# ============================================================================
# KNOWN OPTIMAL VALUES
# ============================================================================

# Verified optimal energies for small N (from brute-force and literature)
KNOWN_OPTIMA = {
    3: 1,    # [1, -1, 1] or equivalents
    4: 2,    # Multiple optimal sequences
    5: 2,    # Multiple optimal sequences
    6: 7,    # Verified by brute-force
    7: 3,    # Verified by brute-force
    8: 8,    # Verified by brute-force
    11: 2,   # Barker code
    13: 4,   # Barker code
}


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestEnergyFunction:
    """Tests for compute_energy() correctness"""
    
    def test_energy_n3_known(self):
        """Test hand-calculated energy for N=3"""
        # s = [1, -1, 1]
        # C_1 = s[0]*s[1] + s[1]*s[2] = 1*(-1) + (-1)*1 = -2
        # C_2 = s[0]*s[2] = 1*1 = 1
        # E = C_1^2 + C_2^2 = 4 + 1 = 5
        seq = np.array([1, -1, 1])
        assert compute_energy(seq) == 5, f"Expected 5, got {compute_energy(seq)}"
    
    def test_energy_n3_optimal(self):
        """Test optimal energy for N=3"""
        # Optimal for N=3 is E=1
        # s = [1, 1, -1]: C_1 = 1*1 + 1*(-1) = 0, C_2 = 1*(-1) = -1, E = 0 + 1 = 1
        seq = np.array([1, 1, -1])
        assert compute_energy(seq) == 1, f"Expected 1, got {compute_energy(seq)}"
    
    def test_energy_n4_known(self):
        """Test hand-calculated energy for N=4"""
        # s = [1, 1, -1, 1]
        # C_1 = 1 + (-1) + (-1) = -1
        # C_2 = -1 + -1 = -2  (wait, let me recalculate)
        # C_1 = s[0]*s[1] + s[1]*s[2] + s[2]*s[3] = 1 + (-1) + (-1) = -1
        # C_2 = s[0]*s[2] + s[1]*s[3] = -1 + 1 = 0
        # C_3 = s[0]*s[3] = 1
        # E = 1 + 0 + 1 = 2
        seq = np.array([1, 1, -1, 1])
        energy = compute_energy(seq)
        assert energy == 2, f"Expected 2, got {energy}"
    
    def test_energy_nonnegative(self):
        """Energy must always be non-negative (sum of squares)"""
        for _ in range(100):
            N = np.random.randint(3, 15)
            seq = np.random.choice([1, -1], size=N)
            energy = compute_energy(seq)
            assert energy >= 0, f"Energy {energy} is negative for N={N}"
    
    def test_energy_integer(self):
        """Energy must be an integer for ±1 sequences"""
        for _ in range(100):
            N = np.random.randint(3, 15)
            seq = np.random.choice([1, -1], size=N)
            energy = compute_energy(seq)
            assert isinstance(energy, (int, np.integer)), f"Energy {energy} is not integer"


class TestSymmetries:
    """Tests for LABS symmetry properties"""
    
    def test_negation_symmetry(self):
        """E(s) = E(-s) for all sequences"""
        for _ in range(100):
            N = np.random.randint(3, 20)
            seq = np.random.choice([1, -1], size=N)
            assert compute_energy(seq) == compute_energy(-seq), \
                f"Negation symmetry violated for {seq}"
    
    def test_reflection_symmetry(self):
        """E(s) = E(s[::-1]) for all sequences"""
        for _ in range(100):
            N = np.random.randint(3, 20)
            seq = np.random.choice([1, -1], size=N)
            assert compute_energy(seq) == compute_energy(seq[::-1]), \
                f"Reflection symmetry violated for {seq}"
    
    def test_combined_symmetry(self):
        """E(s) = E(-s[::-1]) for all sequences"""
        for _ in range(100):
            N = np.random.randint(3, 20)
            seq = np.random.choice([1, -1], size=N)
            assert compute_energy(seq) == compute_energy(-seq[::-1]), \
                f"Combined symmetry violated for {seq}"
    
    def test_all_symmetries_consistent(self):
        """All 4 symmetric variants have same energy"""
        for _ in range(50):
            N = np.random.randint(5, 15)
            seq = np.random.choice([1, -1], size=N)
            
            e1 = compute_energy(seq)
            e2 = compute_energy(-seq)
            e3 = compute_energy(seq[::-1])
            e4 = compute_energy(-seq[::-1])
            
            assert e1 == e2 == e3 == e4, \
                f"Symmetry inconsistency: {e1}, {e2}, {e3}, {e4}"


class TestKnownOptima:
    """Tests against known optimal energies"""
    
    def test_optimal_n3(self):
        """Verify optimal for N=3"""
        opt_energy, opt_seq = brute_force_optimal(3)
        assert opt_energy == KNOWN_OPTIMA[3], \
            f"N=3 optimal: expected {KNOWN_OPTIMA[3]}, got {opt_energy}"
    
    def test_optimal_n4(self):
        """Verify optimal for N=4"""
        opt_energy, opt_seq = brute_force_optimal(4)
        assert opt_energy == KNOWN_OPTIMA[4], \
            f"N=4 optimal: expected {KNOWN_OPTIMA[4]}, got {opt_energy}"
    
    def test_optimal_n5(self):
        """Verify optimal for N=5"""
        opt_energy, opt_seq = brute_force_optimal(5)
        assert opt_energy == KNOWN_OPTIMA[5], \
            f"N=5 optimal: expected {KNOWN_OPTIMA[5]}, got {opt_energy}"
    
    def test_optimal_n6(self):
        """Verify optimal for N=6"""
        opt_energy, opt_seq = brute_force_optimal(6)
        assert opt_energy == KNOWN_OPTIMA[6], \
            f"N=6 optimal: expected {KNOWN_OPTIMA[6]}, got {opt_energy}"
    
    def test_optimal_n7(self):
        """Verify optimal for N=7"""
        opt_energy, opt_seq = brute_force_optimal(7)
        assert opt_energy == KNOWN_OPTIMA[7], \
            f"N=7 optimal: expected {KNOWN_OPTIMA[7]}, got {opt_energy}"
    
    def test_optimal_n8(self):
        """Verify optimal for N=8"""
        opt_energy, opt_seq = brute_force_optimal(8)
        assert opt_energy == KNOWN_OPTIMA[8], \
            f"N=8 optimal: expected {KNOWN_OPTIMA[8]}, got {opt_energy}"
    
    def test_barker_n11(self):
        """Verify Barker code for N=11"""
        # Known Barker sequence for N=11
        barker_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
        energy = compute_energy(barker_11)
        assert energy == KNOWN_OPTIMA[11], \
            f"Barker N=11: expected {KNOWN_OPTIMA[11]}, got {energy}"
    
    def test_barker_n13(self):
        """Verify Barker code for N=13"""
        # Known Barker sequence for N=13
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        energy = compute_energy(barker_13)
        assert energy == KNOWN_OPTIMA[13], \
            f"Barker N=13: expected {KNOWN_OPTIMA[13]}, got {energy}"


class TestAutocorrelation:
    """Tests for autocorrelation computation"""
    
    def test_autocorrelation_length(self):
        """Autocorrelation should have N-1 values"""
        for N in range(3, 15):
            seq = np.random.choice([1, -1], size=N)
            autocorr = compute_autocorrelation(seq)
            assert len(autocorr) == N - 1, \
                f"Expected {N-1} autocorr values, got {len(autocorr)}"
    
    def test_autocorrelation_range(self):
        """C_k should be in range [-(N-k), N-k]"""
        for _ in range(50):
            N = np.random.randint(5, 15)
            seq = np.random.choice([1, -1], size=N)
            autocorr = compute_autocorrelation(seq)
            
            for k, C_k in enumerate(autocorr, start=1):
                max_val = N - k
                assert -max_val <= C_k <= max_val, \
                    f"C_{k} = {C_k} out of range [-{max_val}, {max_val}]"
    
    def test_energy_from_autocorrelation(self):
        """Energy should equal sum of squared autocorrelations"""
        for _ in range(50):
            N = np.random.randint(5, 15)
            seq = np.random.choice([1, -1], size=N)
            
            autocorr = compute_autocorrelation(seq)
            energy_from_autocorr = sum(c**2 for c in autocorr)
            energy_direct = compute_energy(seq)
            
            assert energy_from_autocorr == energy_direct, \
                f"Energy mismatch: {energy_from_autocorr} vs {energy_direct}"


class TestInteractions:
    """Tests for G2, G4 interaction indices"""
    
    def test_g2_valid_indices(self):
        """All G2 pairs should have valid indices 0 <= i < j < N"""
        for N in range(3, 15):
            G2, _ = get_interactions(N)
            for pair in G2:
                i, j = pair
                assert 0 <= i < j < N, \
                    f"Invalid G2 pair {pair} for N={N}"
    
    def test_g4_valid_indices(self):
        """All G4 quads should have 4 distinct valid indices"""
        for N in range(4, 12):
            _, G4 = get_interactions(N)
            for quad in G4:
                assert len(quad) == 4, f"G4 quad has {len(quad)} elements"
                assert len(set(quad)) == 4, f"G4 quad {quad} has duplicates"
                for idx in quad:
                    assert 0 <= idx < N, \
                        f"Invalid G4 index {idx} in {quad} for N={N}"
    
    def test_g2_count_formula(self):
        """
        G2 count should match: sum_{k=1}^{N-1} (N-k) = N(N-1)/2
        """
        for N in range(3, 20):
            G2, _ = get_interactions(N)
            expected = N * (N - 1) // 2
            assert len(G2) == expected, \
                f"G2 count for N={N}: expected {expected}, got {len(G2)}"
    
    def test_g4_no_duplicates(self):
        """G4 should have no duplicate quadruples"""
        for N in range(4, 12):
            _, G4 = get_interactions(N)
            # Convert to tuples for set comparison
            G4_tuples = [tuple(q) for q in G4]
            assert len(G4_tuples) == len(set(G4_tuples)), \
                f"G4 has duplicate quadruples for N={N}"


class TestMTS:
    """Tests for Memetic Tabu Search algorithm"""
    
    def test_mts_improves_or_maintains(self):
        """Tabu search should never make energy worse overall"""
        for _ in range(20):
            N = np.random.randint(5, 15)
            initial_seq = np.random.choice([1, -1], size=N)
            initial_energy = compute_energy(initial_seq)
            
            final_seq, final_energy = tabu_search(initial_seq, max_iter=50)
            
            assert final_energy <= initial_energy, \
                f"MTS worsened energy: {initial_energy} -> {final_energy}"
    
    def test_mts_finds_known_n5(self):
        """MTS should find optimal or near-optimal for N=5"""
        successes = 0
        trials = 10
        
        for _ in range(trials):
            initial = np.random.choice([1, -1], size=5)
            final_seq, final_energy = tabu_search(initial, max_iter=100)
            
            if final_energy <= KNOWN_OPTIMA[5] + 2:  # Allow small gap
                successes += 1
        
        assert successes >= trials // 2, \
            f"MTS found good solution only {successes}/{trials} times for N=5"
    
    def test_mts_finds_known_n7(self):
        """MTS should find optimal or near-optimal for N=7"""
        successes = 0
        trials = 10
        
        for _ in range(trials):
            initial = np.random.choice([1, -1], size=7)
            final_seq, final_energy = tabu_search(initial, max_iter=100)
            
            if final_energy <= KNOWN_OPTIMA[7] + 4:  # Allow small gap
                successes += 1
        
        assert successes >= trials // 2, \
            f"MTS found good solution only {successes}/{trials} times for N=7"
    
    def test_mts_output_valid(self):
        """MTS output should be valid ±1 sequence"""
        for _ in range(20):
            N = np.random.randint(5, 15)
            initial = np.random.choice([1, -1], size=N)
            final_seq, final_energy = tabu_search(initial)
            
            assert len(final_seq) == N, f"Output length {len(final_seq)} != {N}"
            assert set(final_seq).issubset({1, -1}), \
                f"Output contains non-±1 values: {set(final_seq)}"
            assert compute_energy(final_seq) == final_energy, \
                "Returned energy doesn't match sequence"


class TestPopulationInit:
    """Tests for population initialization"""
    
    def test_random_population_size(self):
        """Random population should have correct size"""
        for _ in range(10):
            N = np.random.randint(5, 20)
            pop_size = np.random.randint(5, 30)
            pop = initialize_population_random(N, pop_size)
            
            assert len(pop) == pop_size, \
                f"Population size {len(pop)} != {pop_size}"
    
    def test_random_population_sequence_length(self):
        """Each sequence in population should have length N"""
        N, pop_size = 10, 20
        pop = initialize_population_random(N, pop_size)
        
        for i, seq in enumerate(pop):
            assert len(seq) == N, \
                f"Sequence {i} has length {len(seq)} != {N}"
    
    def test_random_population_values(self):
        """All sequences should contain only ±1"""
        N, pop_size = 10, 20
        pop = initialize_population_random(N, pop_size)
        
        for i, seq in enumerate(pop):
            assert set(seq).issubset({1, -1}), \
                f"Sequence {i} contains invalid values"


# ============================================================================
# BENCHMARK TESTS (Optional - for performance measurement)
# ============================================================================

class TestBenchmarks:
    """Performance benchmarks (not correctness tests)"""
    
    def test_energy_performance(self):
        """Benchmark energy computation time"""
        N_values = [10, 20, 30, 50]
        times = {}
        
        for N in N_values:
            seq = np.random.choice([1, -1], size=N)
            
            start = time.perf_counter()
            for _ in range(1000):
                compute_energy(seq)
            elapsed = time.perf_counter() - start
            
            times[N] = elapsed / 1000  # Average time per call
        
        print(f"\n  Energy computation times (CPU/NumPy):")
        for N, t in times.items():
            print(f"    N={N}: {t*1e6:.2f} µs")
        
        # Just ensure it completes
        assert True
    
    def test_mts_performance(self):
        """Benchmark MTS time to solution"""
        N_values = [10, 15, 20]
        times = {}
        
        for N in N_values:
            trial_times = []
            for _ in range(3):
                initial = np.random.choice([1, -1], size=N)
                
                start = time.perf_counter()
                tabu_search(initial, max_iter=100)
                elapsed = time.perf_counter() - start
                
                trial_times.append(elapsed)
            
            times[N] = np.mean(trial_times)
        
        print(f"\n  MTS time to solution (100 iterations, CPU):")
        for N, t in times.items():
            print(f"    N={N}: {t*1000:.2f} ms")
        
        assert True


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests without pytest"""
    test_classes = [
        TestEnergyFunction,
        TestSymmetries,
        TestKnownOptima,
        TestAutocorrelation,
        TestInteractions,
        TestMTS,
        TestPopulationInit,
        TestBenchmarks,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    print("=" * 70)
    print("LABS Test Suite - Milestone 3 Step A: CPU Validation")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}")
                    print(f"    Error: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"  ✗ {method_name}")
                    print(f"    Exception: {type(e).__name__}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error[:50]}...")
        return 1
    else:
        print("\n✓ All tests passed! CPU validation complete.")
        return 0


if __name__ == "__main__":
    # Run standalone if not using pytest
    sys.exit(run_all_tests())
