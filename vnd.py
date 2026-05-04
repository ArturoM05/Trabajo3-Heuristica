"""
VND - Variable Neighborhood Descent para NWJSSP
=================================================
Implementa el pseudocódigo estándar de VND con tres vecindarios:

    N1: 2-opt      → swap de trabajos CONSECUTIVOS
    N2: Swap-range → swap con distancia ≤ max_range
    N3: Insertion  → inserción con distancia ≤ max_range

Cada vecindario puede usar cualquiera de las dos estrategias de mejora:
    "BI"  Best Improvement  → evalúa TODOS los vecinos de Nj y aplica el mejor
    "FI"  First Improvement → acepta el PRIMER vecino que mejore y para

Parámetros del VNDSearch:
    max_range   : distancia máxima para N2 (Swap) y N3 (Insertion). Default 10.
    improve_n1  : "BI" | "FI"  para el vecindario 2-opt.      Default "BI".
    improve_n2  : "BI" | "FI"  para el vecindario Swap-range. Default "BI".
    improve_n3  : "BI" | "FI"  para el vecindario Insertion.  Default "FI".
    time_limit  : segundos máximos totales. Default 3600 (1 h).

Pseudocódigo seguido:
    s ← initial_solution()
    j = 1
    while j <= number_of_neighborhoods:
        Find s' ∈ Nj(s)
        if f(s') < f(s):  j = 1 ; s ← s'
        else:             j = j + 1
    return s
"""

import time
from constructive import ConstructiveAlgorithm
from read_instances import calculate_flow_time

# Tiempo máximo (s) por llamada interna de búsqueda local (10 min)
LOCAL_TIME_LIMIT = 600


def evaluate_sequence(job_sequence, constructive_algo):
    """
    Evalúa el flow time de una secuencia usando la lógica del constructivo.
    Usa find_earliest_start_time para permitir overlap entre trabajos en
    máquinas distintas, igual que el constructivo original.

    Args:
        job_sequence: lista de job_ids en el orden a programar
        constructive_algo: instancia de ConstructiveAlgorithm

    Returns:
        flow_time: suma de tiempos de completación
        job_start_times: tiempos de inicio resultantes
    """
    job_start_times = [0] * constructive_algo.n
    machine_schedule = {}

    for job_id in job_sequence:
        start_time = constructive_algo.find_earliest_start_time(job_id, machine_schedule)
        job_start_times[job_id] = start_time
        current_time = start_time
        for machine, processing_time in constructive_algo.operations[job_id]:
            if machine not in machine_schedule:
                machine_schedule[machine] = []
            machine_schedule[machine].append(
                (current_time, current_time + processing_time, job_id)
            )
            current_time += processing_time

    flow_time, _ = calculate_flow_time(
        job_start_times, constructive_algo.operations, constructive_algo.release_dates
    )
    return flow_time, job_start_times

# ===========================================================================
# Vecindario N1 – 2-opt (swap de consecutivos)
# ===========================================================================

def _two_opt_BI(seq, algo, deadline):
    """Best Improvement: evalúa todos los pares consecutivos, aplica el mejor."""
    current_flow, current_starts = evaluate_sequence(seq, algo)
    best_delta, best_i = 0, -1

    for i in range(len(seq) - 1):
        if time.time() >= deadline:
            break
        nb = seq[:]
        nb[i], nb[i + 1] = nb[i + 1], nb[i]
        nb_flow, _ = evaluate_sequence(nb, algo)
        delta = current_flow - nb_flow
        if delta > best_delta:
            best_delta, best_i = delta, i

    if best_i >= 0:
        new_seq = seq[:]
        new_seq[best_i], new_seq[best_i + 1] = new_seq[best_i + 1], new_seq[best_i]
        new_flow, new_starts = evaluate_sequence(new_seq, algo)
        return new_seq, new_flow, new_starts, True

    return seq, current_flow, current_starts, False


def _two_opt_FI(seq, algo, deadline):
    """First Improvement: acepta el primer par consecutivo que mejore."""
    current_flow, current_starts = evaluate_sequence(seq, algo)

    for i in range(len(seq) - 1):
        if time.time() >= deadline:
            break
        nb = seq[:]
        nb[i], nb[i + 1] = nb[i + 1], nb[i]
        nb_flow, nb_starts = evaluate_sequence(nb, algo)
        if nb_flow < current_flow:
            return nb, nb_flow, nb_starts, True

    return seq, current_flow, current_starts, False


# ===========================================================================
# Vecindario N2 – Swap con rango
# ===========================================================================

def _swap_BI(seq, algo, max_range, deadline):
    """Best Improvement: evalúa todos los swaps con distancia ≤ max_range, aplica el mejor."""
    current_flow, current_starts = evaluate_sequence(seq, algo)
    best_delta, best_i, best_j = 0, -1, -1
    n = len(seq)

    for i in range(n - 1):
        if time.time() >= deadline:
            break
        for j in range(i + 1, min(i + max_range + 1, n)):
            if time.time() >= deadline:
                break
            nb = seq[:]
            nb[i], nb[j] = nb[j], nb[i]
            nb_flow, _ = evaluate_sequence(nb, algo)
            delta = current_flow - nb_flow
            if delta > best_delta:
                best_delta, best_i, best_j = delta, i, j

    if best_i >= 0:
        new_seq = seq[:]
        new_seq[best_i], new_seq[best_j] = new_seq[best_j], new_seq[best_i]
        new_flow, new_starts = evaluate_sequence(new_seq, algo)
        return new_seq, new_flow, new_starts, True

    return seq, current_flow, current_starts, False


def _swap_FI(seq, algo, max_range, deadline):
    """First Improvement: acepta el primer swap con distancia ≤ max_range que mejore."""
    current_flow, current_starts = evaluate_sequence(seq, algo)
    n = len(seq)

    for i in range(n - 1):
        if time.time() >= deadline:
            break
        for j in range(i + 1, min(i + max_range + 1, n)):
            if time.time() >= deadline:
                break
            nb = seq[:]
            nb[i], nb[j] = nb[j], nb[i]
            nb_flow, nb_starts = evaluate_sequence(nb, algo)
            if nb_flow < current_flow:
                return nb, nb_flow, nb_starts, True

    return seq, current_flow, current_starts, False


# ===========================================================================
# Vecindario N3 – Insertion con rango
# ===========================================================================

def _insertion_BI(seq, algo, max_range, deadline):
    """Best Improvement: evalúa todas las inserciones con distancia ≤ max_range, aplica la mejor."""
    current_flow, current_starts = evaluate_sequence(seq, algo)
    best_delta, best_seq, best_starts = 0, None, None
    n = len(seq)

    for i in range(n):
        if time.time() >= deadline:
            break
        j_min = max(0, i - max_range)
        j_max = min(n - 1, i + max_range)
        for j in range(j_min, j_max + 1):
            if j == i or time.time() >= deadline:
                continue
            nb = seq[:]
            job = nb.pop(i)
            nb.insert(j if j <= i else j - 1, job)
            nb_flow, nb_starts = evaluate_sequence(nb, algo)
            delta = current_flow - nb_flow
            if delta > best_delta:
                best_delta, best_seq, best_starts = delta, nb, nb_starts

    if best_seq is not None:
        return best_seq, current_flow - best_delta, best_starts, True

    return seq, current_flow, current_starts, False


def _insertion_FI(seq, algo, max_range, deadline):
    """First Improvement: acepta la primera inserción con distancia ≤ max_range que mejore."""
    current_flow, current_starts = evaluate_sequence(seq, algo)
    n = len(seq)

    for i in range(n):
        if time.time() >= deadline:
            break
        j_min = max(0, i - max_range)
        j_max = min(n - 1, i + max_range)
        for j in range(j_min, j_max + 1):
            if j == i or time.time() >= deadline:
                continue
            nb = seq[:]
            job = nb.pop(i)
            nb.insert(j if j <= i else j - 1, job)
            nb_flow, nb_starts = evaluate_sequence(nb, algo)
            if nb_flow < current_flow:
                return nb, nb_flow, nb_starts, True

    return seq, current_flow, current_starts, False


# ===========================================================================
# Dispatcher: elige la función correcta según vecindario y estrategia
# ===========================================================================

_NEIGHBORHOOD_FN = {
    # (vecindario, estrategia) → función
    (1, "BI"): lambda seq, algo, r, dl: _two_opt_BI(seq, algo, dl),
    (1, "FI"): lambda seq, algo, r, dl: _two_opt_FI(seq, algo, dl),
    (2, "BI"): lambda seq, algo, r, dl: _swap_BI(seq, algo, r, dl),
    (2, "FI"): lambda seq, algo, r, dl: _swap_FI(seq, algo, r, dl),
    (3, "BI"): lambda seq, algo, r, dl: _insertion_BI(seq, algo, r, dl),
    (3, "FI"): lambda seq, algo, r, dl: _insertion_FI(seq, algo, r, dl),
}


# ===========================================================================
# Clase principal VNDSearch
# ===========================================================================

class VNDSearch:
    """
    Variable Neighborhood Descent con tres vecindarios parametrizables.

    Parámetros
    ----------
    max_range  : int   distancia máxima para N2-Swap y N3-Insertion (default 10)
    improve_n1 : str   "BI" o "FI" para el vecindario 2-opt      (default "BI")
    improve_n2 : str   "BI" o "FI" para el vecindario Swap-range (default "BI")
    improve_n3 : str   "BI" o "FI" para el vecindario Insertion  (default "FI")
    time_limit : float segundos máximos totales                   (default 3600)
    """

    def __init__(self, n, m, operations, release_dates,
                 max_range: int = 10,
                 improve_n1: str = "BI",
                 improve_n2: str = "BI",
                 improve_n3: str = "FI",
                 time_limit: float = 3600.0):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates
        self.max_range = max_range
        self.improve_n1 = improve_n1.upper()
        self.improve_n2 = improve_n2.upper()
        self.improve_n3 = improve_n3.upper()
        self.time_limit = time_limit
        self._algo = ConstructiveAlgorithm(n, m, operations, release_dates)

        # Validar
        for name, val in [("improve_n1", self.improve_n1),
                          ("improve_n2", self.improve_n2),
                          ("improve_n3", self.improve_n3)]:
            if val not in ("BI", "FI"):
                raise ValueError(f"{name} debe ser 'BI' o 'FI', recibido: '{val}'")

    def _improvement_for(self, j):
        return [self.improve_n1, self.improve_n2, self.improve_n3][j - 1]

    def _apply_neighborhood(self, j, seq, global_deadline):
        """Aplica una pasada del vecindario j (1-indexed) con su estrategia configurada."""
        local_deadline = min(global_deadline, time.time() + LOCAL_TIME_LIMIT)
        strategy = self._improvement_for(j)
        fn = _NEIGHBORHOOD_FN[(j, strategy)]
        return fn(seq, self._algo, self.max_range, local_deadline)

    def solve(self, initial_solution=None):
        """
        Ejecuta VND siguiendo el pseudocódigo estándar.

        Returns
        -------
        job_start_times  : list[int]  tiempos de inicio de cada trabajo
        flow_time        : float      suma de tiempos de completación
        computation_time : float      tiempo de cómputo en milisegundos
        """
        start_computation = time.time()
        global_deadline = start_computation + self.time_limit

        if initial_solution is None:
            initial_solution, _, _ = self._algo.solve()

        seq = sorted(range(self.n), key=lambda jj: initial_solution[jj])
        current_flow, current_starts = evaluate_sequence(seq, self._algo)

        j = 1
        while j <= 3 and time.time() < global_deadline:
            new_seq, new_flow, new_starts, improved = self._apply_neighborhood(
                j, seq, global_deadline
            )
            if improved:
                seq, current_flow, current_starts = new_seq, new_flow, new_starts
                j = 1
            else:
                j += 1

        computation_time = (time.time() - start_computation) * 1000
        return current_starts, current_flow, computation_time