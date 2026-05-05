"""
ILS-Tabú con Memoria de Frecuencia para NWJSSP
================================================
Combina tres elementos del enunciado:

1. PERTURBACIÓN / MUTACIÓN  (ILS)
   En cada iteración se perturba la solución con un double-bridge adaptado:
   se extraen k trabajos al azar y se reinsertan en posiciones distintas.

2. MEMORIA DE CORTO PLAZO – Lista Tabú (recencia)
   Prohíbe repetir los últimos tabu_tenure movimientos (pares de posiciones
   intercambiadas). Criterio de aspiración: se permite si supera el mejor global.

3. MEMORIA DE LARGO PLAZO – Frecuencia
   Cuenta cuántas veces cada trabajo fue desplazado de su posición.
   Tras diversification_trigger iteraciones sin mejora global se generan
   n_diverse_candidates perturbaciones sesgadas hacia los trabajos más
   frecuentes y se elige la que produce mejor resultado tras búsqueda local.

Búsqueda local interna: VND (N1=2-opt-BI, N2=Swap-BI, N3=Insertion-FI)
con tope de LOCAL_TIME_LIMIT segundos POR PASADA COMPLETA del VND,
lo que garantiza múltiples iteraciones del ILS dentro de la hora.
"""

import time
import random
from constructive import ConstructiveAlgorithm
from vnd import (evaluate_sequence,
                 _two_opt_BI, _two_opt_FI,
                 _swap_BI,    _swap_FI,
                 _insertion_BI, _insertion_FI)

# Tope de tiempo por llamada a búsqueda local interna.
# Valor conservador para que el ILS pueda iterar decenas de veces en 1 hora.
LOCAL_TIME_LIMIT = 60   # 1 minuto por llamada a _local_search


# ---------------------------------------------------------------------------
# Búsqueda local interna: VND (N1→N2→N3, estrategias BI/BI/FI por defecto)
# ---------------------------------------------------------------------------

def _local_search(job_sequence, algo, max_range, deadline,
                  improve_n1="BI", improve_n2="BI", improve_n3="FI"):
    """
    VND interno: cicla por N1=2-opt, N2=Swap-range, N3=Insertion-range
    hasta que ninguno mejora o se alcanza deadline.
    Retorna (secuencia, flow_time, starts, n_calls) donde n_calls es el
    número de evaluaciones de vecindario realizadas.
    """
    _N1 = _two_opt_BI  if improve_n1 == "BI" else _two_opt_FI
    _N2 = _swap_BI     if improve_n2 == "BI" else _swap_FI
    _N3 = _insertion_BI if improve_n3 == "BI" else _insertion_FI

    n_calls = 0
    j = 1
    while j <= 3 and time.time() < deadline:
        if j == 1:
            new_seq, new_flow, new_starts, improved = _N1(job_sequence, algo, deadline)
        elif j == 2:
            new_seq, new_flow, new_starts, improved = _N2(job_sequence, algo, max_range, deadline)
        else:
            new_seq, new_flow, new_starts, improved = _N3(job_sequence, algo, max_range, deadline)

        n_calls += 1
        if improved:
            job_sequence = new_seq
            j = 1
        else:
            j += 1

    flow, starts = evaluate_sequence(job_sequence, algo)
    return job_sequence, flow, starts, n_calls


# ---------------------------------------------------------------------------
# Perturbación (ILS): double-bridge adaptado a permutaciones lineales
# ---------------------------------------------------------------------------

def _perturb(job_sequence, k, rng):
    """
    Extrae k trabajos en posiciones aleatorias y los reinserta en
    posiciones distintas. k se clampea a [2, n//2].
    Retorna (nueva_secuencia, set_de_posiciones_modificadas).
    """
    n = len(job_sequence)
    k = max(2, min(k, n // 2))
    seq = job_sequence[:]

    positions = sorted(rng.sample(range(n), k))
    jobs_out  = [seq[p] for p in positions]

    # Eliminar extraídos
    remaining = [j for j in seq if j not in set(jobs_out)]

    # Reinsertar en posiciones nuevas
    for job in jobs_out:
        pos = rng.randint(0, len(remaining))
        remaining.insert(pos, job)

    # Registrar qué posiciones cambiaron
    changed_pos = {i for i in range(n) if remaining[i] != job_sequence[i]}
    return remaining, changed_pos


# ---------------------------------------------------------------------------
# Perturbación sesgada (diversificación por frecuencia)
# ---------------------------------------------------------------------------

def _perturb_biased(job_sequence, freq_counter, k, rng):
    """
    Como _perturb pero los trabajos a extraer se eligen con probabilidad
    proporcional a su frecuencia acumulada (los más movidos tienen más
    probabilidad de ser perturbados → fuerza diversificación).
    """
    n = len(job_sequence)
    k = max(2, min(k, n // 2))
    seq = job_sequence[:]

    weights = [freq_counter[j] + 1 for j in seq]   # +1 para evitar peso 0
    total   = sum(weights)
    probs   = [w / total for w in weights]

    chosen, pool, pool_p = [], list(range(n)), probs[:]
    for _ in range(k):
        if not pool:
            break
        r, cum = rng.random(), 0.0
        for idx, p in enumerate(pool_p):
            cum += p
            if r <= cum:
                chosen.append(pool[idx])
                pool.pop(idx); pool_p.pop(idx)
                s = sum(pool_p)
                if s > 0:
                    pool_p = [pp / s for pp in pool_p]
                break

    jobs_out  = [seq[p] for p in sorted(chosen)]
    remaining = [j for j in seq if j not in set(jobs_out)]
    for job in jobs_out:
        remaining.insert(rng.randint(0, len(remaining)), job)

    changed_pos = {i for i in range(n) if remaining[i] != job_sequence[i]}
    return remaining, changed_pos


# ---------------------------------------------------------------------------
# Lista Tabú  (basada en recencia, sobre pares de posiciones)
# ---------------------------------------------------------------------------

class TabuList:
    """
    Almacena los pares de posiciones (i, j) intercambiadas en los últimos
    `tenure` movimientos aceptados. Un movimiento es tabú si su par de
    posiciones aparece en la lista.
    """

    def __init__(self, tenure: int):
        self.tenure = tenure
        self._entries = []   # [(pos_i, pos_j), ...]

    def is_tabu(self, move):
        return move in self._entries

    def add(self, move):
        self._entries.append(move)
        if len(self._entries) > self.tenure:
            self._entries.pop(0)

    def clear(self):
        self._entries.clear()


def _extract_tabu_move(seq_before, seq_after):
    """
    Compara dos secuencias y devuelve el par de posiciones que cambiaron
    como tupla ordenada (min_pos, max_pos), o None si no hay exactamente 2.
    """
    diffs = [i for i in range(len(seq_before)) if seq_before[i] != seq_after[i]]
    if len(diffs) >= 2:
        return (diffs[0], diffs[-1])
    return None


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ILSTabuSearch:
    """
    ILS + Lista Tabú + Memoria de Frecuencia para NWJSSP.

    Parámetros
    ----------
    max_range              : rango máximo para Swap e Insertion   (default 10)
    time_limit             : tiempo total máximo en segundos       (default 3600)
    tabu_tenure            : longitud de la lista tabú             (default 15)
    perturbation_k         : trabajos a mover en cada perturbación (default 4)
    diversification_trigger: iteraciones sin mejora antes de       (default 20)
                             diversificar con memoria de frecuencia
    n_diverse_candidates   : candidatos en la fase de diversif.    (default 5)
    seed                   : semilla aleatoria                     (default 42)
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

    def solve(self, initial_solution=None):
        """
        Ejecuta ILS-Tabú con memoria de frecuencia.

        Returns
        -------
        job_start_times  : tiempos de inicio de cada trabajo
        flow_time        : suma de tiempos de completación
        computation_time : tiempo de cómputo en milisegundos
        ls_calls         : total de llamadas a búsquedas locales internas
        """
        start_t = time.time()
        deadline = start_t + self.time_limit

        # ── Solución inicial ──────────────────────────────────────────
        if initial_solution is None:
            initial_solution, _, _ = self._algo.solve()

        seq = sorted(range(self.n), key=lambda jj: initial_solution[jj])

        # Búsqueda local inicial (tope = LOCAL_TIME_LIMIT)
        ls_dl = min(deadline, time.time() + LOCAL_TIME_LIMIT)
        seq, cur_flow, cur_starts, calls = _local_search(
            seq, self._algo, self.max_range, ls_dl
        )
        ls_calls = calls

        best_seq, best_flow, best_starts = seq[:], cur_flow, cur_starts

        # ── Estructuras de memoria ────────────────────────────────────
        tabu = TabuList(self.tabu_tenure)
        # freq_counter[job_id] = nº de veces que ese trabajo fue desplazado
        freq_counter = {j: 0 for j in range(self.n)}
        no_improve = 0   # iteraciones sin mejora del mejor global

        # ── Bucle principal ILS ───────────────────────────────────────
        while time.time() < deadline:

            # ── DIVERSIFICACIÓN (memoria de largo plazo) ──────────────
            if no_improve >= self.diversification_trigger:
                remaining_t = deadline - time.time()
                # Repartir el tiempo disponible entre los candidatos
                ls_budget = min(LOCAL_TIME_LIMIT,
                                remaining_t / (self.n_diverse_candidates + 1))

                best_cand_seq, best_cand_flow = None, float('inf')

                for _ in range(self.n_diverse_candidates):
                    if time.time() >= deadline:
                        break
                    cand, _ = _perturb_biased(
                        best_seq, freq_counter, self.perturbation_k, self.rng
                    )
                    cand_dl = min(deadline, time.time() + ls_budget)
                    cand, cand_flow, _, c = _local_search(
                        cand, self._algo, self.max_range, cand_dl
                    )
                    ls_calls += c
                    if cand_flow < best_cand_flow:
                        best_cand_flow, best_cand_seq = cand_flow, cand[:]

                if best_cand_seq is not None:
                    seq      = best_cand_seq
                    cur_flow, cur_starts = evaluate_sequence(seq, self._algo)
                    tabu.clear()
                    no_improve = 0

                    if cur_flow < best_flow:
                        best_flow, best_seq, best_starts = cur_flow, seq[:], cur_starts
                continue   # reiniciar loop tras diversificación

            # ── PERTURBACIÓN (ILS) ────────────────────────────────────
            prev_seq = seq[:]
            perturbed, changed_pos = _perturb(seq, self.perturbation_k, self.rng)

            # Actualizar frecuencia: incrementar contador de cada trabajo
            # que cambió de posición
            for pos in changed_pos:
                freq_counter[perturbed[pos]] += 1

            # ── BÚSQUEDA LOCAL ────────────────────────────────────────
            ls_dl = min(deadline, time.time() + LOCAL_TIME_LIMIT)
            new_seq, new_flow, new_starts, c = _local_search(
                perturbed, self._algo, self.max_range, ls_dl
            )
            ls_calls += c

            # ── CRITERIO DE ACEPTACIÓN (tabú + aspiración) ────────────
            tabu_move = _extract_tabu_move(prev_seq, new_seq)
            is_tabu   = (tabu_move is not None) and tabu.is_tabu(tabu_move)
            aspiration = new_flow < best_flow

            if (not is_tabu) or aspiration:
                seq, cur_flow, cur_starts = new_seq, new_flow, new_starts

                if tabu_move is not None:
                    tabu.add(tabu_move)

                if cur_flow < best_flow:
                    best_flow, best_seq, best_starts = cur_flow, seq[:], cur_starts
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1   # movimiento tabú descartado

        comp_time = (time.time() - start_t) * 1000
        return best_starts, best_flow, comp_time, ls_calls