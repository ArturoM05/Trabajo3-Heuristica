"""
ILS-Tabú con Memoria de Frecuencia para NWJSSP
================================================
Metaheurístico que combina tres elementos del enunciado:

1. PERTURBACIÓN / MUTACIÓN  (ILS – Iterated Local Search)
   - En cada iteración se perturba la solución actual con un double-bridge
     adaptado a permutaciones lineales: se extraen k trabajos al azar y se
     reinsertan en posiciones aleatorias distintas. Esto escapa de óptimos
     locales sin destruir demasiado la solución.

2. MEMORIA DE CORTO PLAZO – Lista Tabú (basada en recencia)
   - Se prohíbe revertir movimientos recientes. La lista tabú almacena los
     pares (job_i, job_j) de los últimos tabu_tenure iteraciones.
   - Un movimiento swap (i,j) es tabú si el par de trabajos en esas
     posiciones aparece en la lista. Se permite igualmente si el vecino
     supera la mejor solución global (criterio de aspiración).

3. MEMORIA DE LARGO PLAZO – Frecuencia
   - Se lleva un contador de cuántas veces cada trabajo ha sido movido.
   - Cada cierto número de iteraciones sin mejora global
     (diversification_trigger), se aplica diversificación: se generan
     múltiples perturbaciones sesgadas hacia trabajos poco frecuentes y
     se elige la que produce mejor flow_time tras una búsqueda local rápida.

Búsqueda local interna: VND ligero (N1=2-opt, N2=Swap-10) con tiempo
máximo LOCAL_TIME_LIMIT segundos para no consumir todo el presupuesto en
una sola iteración.

Criterio de parada: tiempo límite configurable (por defecto 3600 s = 1 h).
"""

import time
import random
from constructive import ConstructiveAlgorithm
from vnd import evaluate_sequence
from vnd import _two_opt_FI as _one_pass_two_opt
from vnd import _swap_FI as _one_pass_swap
from vnd import _insertion_FI as _one_pass_insertion

# Tiempo máximo (s) por búsqueda local interna
LOCAL_TIME_LIMIT = 600  # 10 minutos


# ---------------------------------------------------------------------------
# Búsqueda local interna: VND ligero (N1 + N2 + N3)
# ---------------------------------------------------------------------------

def _local_search(job_sequence, algo, max_range, deadline):
    """
    VND interno con N1=2-opt, N2=Swap-range, N3=Insertion-range.
    Se detiene cuando ningún vecindario mejora o se alcanza el deadline.
    """
    current_flow, current_starts = evaluate_sequence(job_sequence, algo)
    j = 1
    while j <= 3 and time.time() < deadline:
        if j == 1:
            new_seq, new_flow, new_starts, improved = _one_pass_two_opt(
                job_sequence, algo, deadline
            )
        elif j == 2:
            new_seq, new_flow, new_starts, improved = _one_pass_swap(
                job_sequence, algo, max_range, deadline
            )
        else:
            new_seq, new_flow, new_starts, improved = _one_pass_insertion(
                job_sequence, algo, max_range, deadline
            )

        if improved:
            job_sequence = new_seq
            current_flow = new_flow
            current_starts = new_starts
            j = 1
        else:
            j += 1

    return job_sequence, current_flow, current_starts


# ---------------------------------------------------------------------------
# Perturbación: double-bridge adaptado a permutaciones (ILS)
# ---------------------------------------------------------------------------

def _perturb(job_sequence, k, rng):
    """
    Extrae k trabajos en posiciones aleatorias y los reinserta en
    posiciones distintas también aleatorias. Garantiza que la secuencia
    resultante es diferente a la original (k ≥ 2).
    """
    n = len(job_sequence)
    k = max(2, min(k, n // 2))
    seq = job_sequence[:]

    positions = rng.sample(range(n), k)
    jobs_extracted = [seq[p] for p in sorted(positions)]

    # Eliminar los trabajos de la secuencia
    remaining = [j for j in seq if j not in jobs_extracted]

    # Insertar en posiciones nuevas aleatorias
    for job in jobs_extracted:
        insert_pos = rng.randint(0, len(remaining))
        remaining.insert(insert_pos, job)

    return remaining


# ---------------------------------------------------------------------------
# Perturbación sesgada por frecuencia (diversificación)
# ---------------------------------------------------------------------------

def _perturb_biased(job_sequence, freq_counter, k, rng):
    """
    Perturbación que prioriza mover los trabajos con mayor frecuencia acumulada
    (los más visitados), forzando diversificación.
    """
    n = len(job_sequence)
    k = max(2, min(k, n // 2))
    seq = job_sequence[:]

    # Pesos inversamente proporcionales a la frecuencia (más frecuente = más probable)
    weights = [freq_counter.get(j, 0) + 1 for j in seq]
    total = sum(weights)
    probs = [w / total for w in weights]

    # Selección ponderada sin reemplazo
    chosen = []
    pool = list(range(n))
    pool_probs = probs[:]
    for _ in range(k):
        if not pool:
            break
        r = rng.random()
        cumulative = 0.0
        for idx, p in enumerate(pool_probs):
            cumulative += p
            if r <= cumulative:
                chosen.append(pool[idx])
                pool.pop(idx)
                pool_probs.pop(idx)
                s = sum(pool_probs)
                if s > 0:
                    pool_probs = [pp / s for pp in pool_probs]
                break

    jobs_extracted = [seq[p] for p in sorted(chosen)]
    remaining = [j for j in seq if j not in jobs_extracted]
    for job in jobs_extracted:
        insert_pos = rng.randint(0, len(remaining))
        remaining.insert(insert_pos, job)

    return remaining


# ---------------------------------------------------------------------------
# Lista tabú
# ---------------------------------------------------------------------------

class TabuList:
    """Lista tabú basada en recencia: almacena pares (job_a, job_b) de los
    últimos `tenure` iteraciones."""

    def __init__(self, tenure: int):
        self.tenure = tenure
        self._list = []          # [(job_a, job_b), ...]  más reciente al final

    def is_tabu(self, move):
        """move es una tupla (job_a, job_b) con job_a <= job_b."""
        return move in self._list

    def add(self, move):
        self._list.append(move)
        if len(self._list) > self.tenure:
            self._list.pop(0)

    def clear(self):
        self._list.clear()


# ---------------------------------------------------------------------------
# Clase principal ILSTabuSearch
# ---------------------------------------------------------------------------

class ILSTabuSearch:
    """
    ILS + Lista Tabú + Memoria de frecuencia para NWJSSP.

    Parámetros:
        max_range             : rango máximo para swap e insertion (default 10)
        time_limit            : tiempo total máximo en segundos (default 3600)
        tabu_tenure           : longitud de la lista tabú (default 15)
        perturbation_k        : trabajos a mover en cada perturbación (default 4)
        diversification_trigger: iteraciones sin mejora global antes de
                                 diversificar con memoria de frecuencia (default 20)
        n_diverse_candidates  : candidatos generados en diversificación (default 5)
        seed                  : semilla aleatoria (default 42)
    """

    def __init__(self, n, m, operations, release_dates,
                 max_range: int = 10,
                 time_limit: float = 3600.0,
                 tabu_tenure: int = 15,
                 perturbation_k: int = 4,
                 diversification_trigger: int = 20,
                 n_diverse_candidates: int = 5,
                 seed: int = 42):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates
        self.max_range = max_range
        self.time_limit = time_limit
        self.tabu_tenure = tabu_tenure
        self.perturbation_k = perturbation_k
        self.diversification_trigger = diversification_trigger
        self.n_diverse_candidates = n_diverse_candidates
        self.rng = random.Random(seed)
        self._algo = ConstructiveAlgorithm(n, m, operations, release_dates)

    # ------------------------------------------------------------------

    def _make_tabu_move(self, seq_before, seq_after):
        """Extrae el par de trabajos intercambiados (para lista tabú)."""
        diffs = [(i, seq_before[i], seq_after[i])
                 for i in range(len(seq_before)) if seq_before[i] != seq_after[i]]
        if len(diffs) >= 2:
            jobs = tuple(sorted([diffs[0][1], diffs[1][1]]))
            return jobs
        return None

    # ------------------------------------------------------------------

    def solve(self, initial_solution=None):
        """
        Ejecuta ILS-Tabú con memoria de frecuencia.

        Returns:
            job_start_times  : tiempos de inicio de cada trabajo
            flow_time        : suma de tiempos de completación
            computation_time : tiempo de cómputo en milisegundos
        """
        start_computation = time.time()
        global_deadline = start_computation + self.time_limit

        # ── Solución inicial ──────────────────────────────────────────
        if initial_solution is None:
            initial_solution, _, _ = self._algo.solve()

        job_sequence = sorted(range(self.n), key=lambda jj: initial_solution[jj])

        # Búsqueda local inicial
        ls_deadline = min(global_deadline, time.time() + LOCAL_TIME_LIMIT)
        job_sequence, current_flow, current_starts = _local_search(
            job_sequence, self._algo, self.max_range, ls_deadline
        )

        best_sequence = job_sequence[:]
        best_flow     = current_flow
        best_starts   = current_starts

        # ── Estructuras de memoria ────────────────────────────────────
        tabu_list = TabuList(self.tabu_tenure)
        freq_counter = {j: 0 for j in range(self.n)}   # memoria de largo plazo

        no_improve_count = 0   # iteraciones sin mejora de la mejor solución global

        # ── Bucle principal ───────────────────────────────────────────
        while time.time() < global_deadline:

            # ── DIVERSIFICACIÓN (memoria de largo plazo) ──────────────
            if no_improve_count >= self.diversification_trigger:
                best_candidate_seq  = None
                best_candidate_flow = float('inf')
                ls_budget = min(LOCAL_TIME_LIMIT,
                                (global_deadline - time.time()) / (self.n_diverse_candidates + 1))

                for _ in range(self.n_diverse_candidates):
                    if time.time() >= global_deadline:
                        break
                    cand = _perturb_biased(
                        best_sequence, freq_counter, self.perturbation_k, self.rng
                    )
                    ls_dl = min(global_deadline, time.time() + ls_budget)
                    cand, cand_flow, _ = _local_search(
                        cand, self._algo, self.max_range, ls_dl
                    )
                    if cand_flow < best_candidate_flow:
                        best_candidate_flow = cand_flow
                        best_candidate_seq  = cand

                if best_candidate_seq is not None:
                    job_sequence  = best_candidate_seq
                    current_flow  = best_candidate_flow
                    current_flow, current_starts = evaluate_sequence(
                        job_sequence, self._algo
                    )
                    tabu_list.clear()   # reiniciar lista tabú tras diversificación
                    no_improve_count = 0

                    if current_flow < best_flow:
                        best_flow     = current_flow
                        best_sequence = job_sequence[:]
                        best_starts   = current_starts
                    continue

            # ── PERTURBACIÓN (ILS) ────────────────────────────────────
            prev_sequence = job_sequence[:]
            perturbed = _perturb(job_sequence, self.perturbation_k, self.rng)

            # Actualizar frecuencia de los trabajos movidos
            moved = set(prev_sequence) - set(
                p for p, q in zip(prev_sequence, perturbed) if p == q
            )
            for job in perturbed:
                if job in moved:
                    freq_counter[job] += 1

            # ── BÚSQUEDA LOCAL tras perturbación ──────────────────────
            ls_deadline = min(global_deadline, time.time() + LOCAL_TIME_LIMIT)
            new_seq, new_flow, new_starts = _local_search(
                perturbed, self._algo, self.max_range, ls_deadline
            )

            # ── CRITERIO DE ACEPTACIÓN (lista tabú + aspiración) ──────
            move = self._make_tabu_move(prev_sequence, new_seq)
            is_tabu = (move is not None) and tabu_list.is_tabu(move)
            aspiration = new_flow < best_flow   # criterio de aspiración

            if (not is_tabu) or aspiration:
                job_sequence  = new_seq
                current_flow  = new_flow
                current_starts = new_starts

                if move is not None:
                    tabu_list.add(move)

                # Actualizar mejor solución global
                if current_flow < best_flow:
                    best_flow     = current_flow
                    best_sequence = job_sequence[:]
                    best_starts   = current_starts
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Movimiento tabú sin criterio de aspiración: descartar
                no_improve_count += 1

        computation_time = (time.time() - start_computation) * 1000
        return best_starts, best_flow, computation_time
