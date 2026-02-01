# Product Requirements Document (PRD)

**Project Name:** Phase1
**Team Name:** Eigencrew
**GitHub Repository:** https://github.com/beetrandahiya/2026-NVIDIA

---

## 1. Team Roles & Responsibilities 

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Prakrisht | [@handle] | [@handle] |
| **GPU Acceleration PIC** (Builder) | Priyanshi | [@handle] | [@handle] |
| **Quality Assurance PIC** (Verifier) | Aarav | [@handle] | [@handle] |
| **Technical Marketing PIC** (Storyteller) | Prakrisht | [@handle] | [@handle] |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** Digitized Counterdiabatic Quantum Optimization (DCQO) with Trotterized evolution
    * We implemented the counteradiabatic quantum algorithm from the Kipu Quantum/NVIDIA paper using CUDA-Q. The circuit uses Trotterized time evolution with:
      - 2-body interaction terms (R_YZ, R_ZY gates) for pairs in set G2
      - 4-body interaction terms (R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY gates) for quadruples in set G4
    * The quantum samples are used to seed a classical Memetic Tabu Search (MTS) for hybrid optimization.

* **Motivation:** 
    * **Metric-driven:** The DCQO approach provides "shortcuts to adiabaticity" by adding counterdiabatic terms that suppress diabatic transitions, enabling faster convergence to ground states with fewer Trotter steps.
    * **Problem-specific:** The LABS Hamiltonian structure (quartic terms from $C_k^2$) naturally decomposes into 2-body and 4-body Pauli terms, making DCQO well-suited for this problem.
    * **Hybrid advantage:** Quantum samples from low-energy regions of the landscape provide diverse, high-quality seeds for classical MTS, improving convergence over purely random initialization.

### Literature Review
* **Reference 1:** "Scaling advantage with quantum-enhanced memetic tabu search for LABS" (Kipu Quantum, UPV/EHU, NVIDIA), arXiv:2511.04553v1
    * **Relevance:** This is our primary reference. It establishes the theoretical framework for DCQO applied to LABS, provides the Hamiltonian decomposition (Eq. 15/B3), and demonstrates scaling advantages for the quantum-enhanced hybrid approach.

* **Reference 2:** "Parallel MTS by JPMorgan Chase", arXiv:2504.00987
    * **Relevance:** Provides insights into parallelizing MTS on GPUs, which informs our acceleration strategy for the classical component.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** 
    * Use CUDA-Q's `nvidia` backend for GPU-accelerated state vector simulation
    * Docker container with `nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.13.0` image for consistent CUDA 13 environment
    * GPU passthrough via `--gpus all` flag for RTX 4070 Ti / A100 access
    * The trotterized circuit implements efficient multi-qubit gate decompositions using CNOT ladders for the 4-body R_YZZZ/R_ZYZZ/R_ZZYZ/R_ZZZY rotations

### Classical Acceleration (MTS)
* **Strategy:** 
    * NumPy-based vectorized energy computation: `E(s) = sum(C_k^2)` where `C_k = np.dot(s[:N-k], s[k:])`
    * Population-based parallelism: Multiple MTS instances can run concurrently with different quantum seeds
    * Tabu search uses efficient 1-flip neighborhood evaluation with aspiration criterion for escaping local minima

### Hardware Targets
* **Dev Environment:** 
    * Windows host with Docker Desktop (WSL2 backend)
    * CUDA-Q Docker container for Linux compatibility (CUDA-Q requires Linux)
    * JupyterLab running inside container at `localhost:8888`
* **Production Environment:** 
    * NVIDIA RTX 4070 Ti (local) or Brev L4/A100 instances for large N benchmarks
    * Target: N=20+ for quantum circuit, N=30+ for full hybrid workflow

---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** Inline validation suite in notebook with automated `run_test()` helper function
* **AI Hallucination Guardrails:** 
    * All AI-generated code validated against hand-calculated results for small N
    * Cross-reference quantum energies against brute-force optimal for N≤8
    * CUDA-Q API usage verified against official documentation (avoiding deprecated `dict(result)` patterns)

### Core Correctness Checks
* **Check 1 (Symmetry):** 
    * Negation symmetry: `compute_energy(s) == compute_energy(-s)` for all sequences
    * Reflection symmetry: `compute_energy(s) == compute_energy(s[::-1])`
    * Combined symmetry: `compute_energy(s) == compute_energy(-s[::-1])`
    * Validated across 100 random sequences with zero violations

* **Check 2 (Ground Truth):**
    * Brute-force verification for N=3 to N=8 against published optimal energies:
      - N=3: E_min=1, N=4: E_min=2, N=5: E_min=2, N=6: E_min=7
      - N=7: E_min=3, N=8: E_min=8, N=11: E_min=2 (Barker), N=13: E_min=4 (Barker)
    * Unit test: `compute_energy([1, -1, 1]) == 5` (hand-calculated)

* **Check 3 (Quantum Circuit):**
    * Total sample counts equal shots_count (probability normalization)
    * All bitstrings have correct length N
    * Energy consistency: both conversion methods (`1 if b=='0' else -1` vs `(-1)**int(b)`) produce identical energies

* **Check 4 (MTS Correctness):**
    * Tabu search always improves or maintains energy (20/20 trials)
    * MTS finds known optima for N=5, 7, 11 within tolerance

* **Check 5 (Interaction Indices):**
    * G2 indices valid: 0 ≤ i < j < N for all pairs
    * G4 indices valid: 4 distinct values, all in range [0, N-1]
    * Count verification: |G2| matches analytical formula

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** 
    * VS Code with GitHub Copilot (Claude Opus 4.5) as the primary AI coding assistant
    * All of the code was rechecked for logic, error handling by the team.

### Success Metrics
* **Metric 1 (Correctness):** 100% pass rate on all 24 self-validation tests
* **Metric 2 (Approximation):** QE-MTS achieves lower mean energy than classical MTS for N=11 across 5 trials
* **Metric 3 (Scale):** Successfully run DCQO circuit for N=10+ qubits with meaningful quantum seed quality

### Visualization Plan
* **Plot 1:** MTS Energy Convergence - Best energy vs. generation for N=13
* **Plot 2:** Final Population Distribution - Histogram of energies in final MTS population  
* **Plot 3:** Autocorrelation Profile - Bar chart of C_k values for best sequence found
* **Plot 4:** QE-MTS vs Classical Comparison - Box plot of energy distributions across trials
* **Plot 5:** Trial-by-trial comparison - Line plot showing energy progression for both methods

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** 
    * **Development Phase:** All development till now done locally using Docker container with local RTX 4070 Ti (no cloud credits consumed)
    * **Docker Strategy:** Use `docker run --rm` flag to automatically clean up containers after use; avoid zombie instances
    * **Safeguards:**
      - Set terminal reminders every 30 minutes to verify no idle instances
      - Test all code locally on CPU/small GPU before scaling to expensive instances
