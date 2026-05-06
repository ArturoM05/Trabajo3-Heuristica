"""
MS-LNS-SA: Multi-Start + Large Neighborhood Search + Simulated Annealing
=========================================================================
Combina tres elementos del enunciado:

1. MÚLTIPLES SOLUCIONES INICIALES (Multi-Start)
   Se generan varias soluciones iniciales distintas permutando aleatoriamente
   el orden de construcción del greedy. Cada una se optimiza con una búsqueda
   local rápida y se conserva la mejor como punto de partida global.

2. DESTRUCCIÓN Y REPARACIÓN (LNS – Large Neighborhood Search)
   En cada iteración se destruye parcialmente la solución actual eliminando
   q trabajos seleccionados (operador de destrucción) y se reparan
   reinsertándolos uno a uno en la mejor posición posible (operador de
   reparación greedy). El tamaño q varía adaptativamente: si no hay mejora
   durante varias iteraciones consecutivas, q aumenta para explorar más lejos.

3. ACEPTACIÓN PROBABILÍSTICA – Recocido Simulado (Simulated Annealing)
   Las soluciones peores que la actual se aceptan con probabilidad
   exp(-Δ/T), donde T se enfría geométricamente. Esto permite escapar de
   óptimos locales sin necesidad de perturbaciones agresivas.

Criterio de parada: tiempo límite configurable (por defecto 3600 s = 1 hora).
"""

import time
import random
import math
from constructive import ConstructiveAlgorithm
from vnd import evaluate_sequence, _two_opt_FI, _swap_FI, _insertion_FI


# ---------------------------------------------------------------------------
# Búsqueda local rápida (una pasada FI de cada vecindario)
# ---------------------------------------------------------------------------

def _quick_local_search(seq, algo, max_range, deadline):
    """
    Una pasada de N1-FI → N2-FI → N3-FI sin reiniciar.
    Rápida para usar como post-procesamiento tras destrucción/reparación.
    Retorna (secuencia, flow, starts, n_soluciones_evaluadas).
    """
    n_eval = 0
    for fn, needs_range in [
        (_two_opt_FI,   False),
        (_swap_FI,      True),
        (_insertion_FI, True),
    ]:
        if time.time() >= deadline:
            break
        if needs_range:
            new_seq, new_flow, new_starts, improved = fn(seq, algo, max_range, deadline)
        else:
            new_seq, new_flow, new_starts, improved = fn(seq, algo, deadline)
        n_eval += 1
        if improved:
            seq = new_seq
    flow, starts = evaluate_sequence(seq, algo)
    return seq, flow, starts, n_eval


# ---------------------------------------------------------------------------
# Generación de soluciones iniciales diversas (Multi-Start)
# ---------------------------------------------------------------------------

def _build_random_solution(algo, rng):
    """
    Construye una solución usando el mismo esquema greedy del constructivo
    pero con un orden de trabajos aleatorio (para diversidad).
    """
    n = algo.n
    # Ordenar con prioridad greedy + ruido aleatorio suave
    order = sorted(range(n), key=lambda j: (
        algo.release_dates[j] + rng.uniform(0, 0.3) * sum(pt for _, pt in algo.operations[j]),
        sum(pt for _, pt in algo.operations[j])
    ))

    job_start_times = [0] * n
    machine_schedule = {}
    for job_id in order:
        st = algo.find_earliest_start_time(job_id, machine_schedule)
        job_start_times[job_id] = st
        cur = st
        for machine, pt in algo.operations[job_id]:
            machine_schedule.setdefault(machine, []).append((cur, cur + pt, job_id))
            cur += pt

    from read_instances import calculate_flow_time
    flow, _ = calculate_flow_time(job_start_times, algo.operations, algo.release_dates)
    return job_start_times, flow


# ---------------------------------------------------------------------------
# Operador de DESTRUCCIÓN
# ---------------------------------------------------------------------------

def _destroy(seq, q, rng):
    """
    Elimina q trabajos de la secuencia.
    Estrategia mixta: mitad aleatoria, mitad los más 'tardíos' (últimas posiciones)
    para favorecer reordenamientos significativos.
    Retorna (secuencia_parcial, lista_de_trabajos_eliminados).
    """
    n = len(seq)
    q = max(2, min(q, n - 1))

    n_random = max(1, q // 2)
    n_tail   = q - n_random

    tail_candidates = seq[max(0, n - q * 2):]   # últimos 2q trabajos
    tail_jobs = rng.sample(tail_candidates, min(n_tail, len(tail_candidates)))
    tail_set  = set(tail_jobs)

    remaining_pool = [j for j in seq if j not in tail_set]
    random_jobs = rng.sample(remaining_pool, min(n_random, len(remaining_pool)))

    removed = list(tail_set) + random_jobs
    partial  = [j for j in seq if j not in set(removed)]
    return partial, removed


# ---------------------------------------------------------------------------
# Operador de REPARACIÓN greedy
# ---------------------------------------------------------------------------

def _repair(partial_seq, removed_jobs, algo, deadline):
    """
    Reinserta los trabajos eliminados uno a uno en la posición que minimice
    el incremento de flow time (greedy de inserción).
    Retorna (secuencia_completa, n_soluciones_evaluadas).
    """
    seq = partial_seq[:]
    n_eval = 0

    for job in removed_jobs:
        if time.time() >= deadline:
            # Si se acaba el tiempo, insertar al final
            seq.append(job)
            continue

        best_flow  = float('inf')
        best_pos   = len(seq)   # insertar al final por defecto

        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            flow, _ = evaluate_sequence(candidate, algo)
            n_eval += 1
            if flow < best_flow:
                best_flow = flow
                best_pos  = pos

        seq.insert(best_pos, job)

    return seq, n_eval


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class MSLNSSearch:
    """
    Multi-Start + LNS (destrucción/reparación) + Simulated Annealing.

    Parámetros
    ----------
    max_range          : rango para búsqueda local interna            (default 10)
    time_limit         : segundos totales                              (default 3600)
    n_starts           : soluciones iniciales del Multi-Start          (default 5)
    q_init             : trabajos eliminados en destrucción inicial     (default 4)
    q_max              : máximo trabajos a eliminar                    (default 10)
    no_improve_q_step  : iteraciones sin mejora antes de subir q       (default 15)
    sa_t0              : temperatura inicial del SA                    (default None → auto)
    sa_cooling         : factor de enfriamiento geométrico             (default 0.995)
    seed               : semilla aleatoria                             (default 42)
    """

    def __init__(self, n, m, operations, release_dates,
                 max_range: int = 10,
                 time_limit: float = 3600.0,
                 n_starts: int = 5,
                 q_init: int = 4,
                 q_max: int = 10,
                 no_improve_q_step: int = 15,
                 sa_t0: float = None,
                 sa_cooling: float = 0.995,
                 seed: int = 42):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates
        self.max_range = max_range
        self.time_limit = time_limit
        self.n_starts = n_starts
        self.q_init = q_init
        self.q_max = q_max
        self.no_improve_q_step = no_improve_q_step
        self.sa_t0 = sa_t0
        self.sa_cooling = sa_cooling
        self.rng = random.Random(seed)
        self._algo = ConstructiveAlgorithm(n, m, operations, release_dates)

    # ------------------------------------------------------------------

    def _multi_start(self, deadline):
        """
        Genera n_starts soluciones diversas, aplica búsqueda local a cada una
        y devuelve la mejor secuencia junto con su flow y starts.
        También retorna el total de soluciones evaluadas en esta fase.
        """
        best_seq, best_flow, best_starts = None, float('inf'), None
        n_eval = 0

        # Tiempo por solución inicial: repartir equitativamente
        budget = min(30.0, (deadline - time.time()) / (self.n_starts + 1))

        for _ in range(self.n_starts):
            if time.time() >= deadline:
                break

            sol0, _ = _build_random_solution(self._algo, self.rng)
            seq = sorted(range(self.n), key=lambda j: sol0[j])

            ls_dl = min(deadline, time.time() + budget)
            seq, flow, starts, ev = _quick_local_search(
                seq, self._algo, self.max_range, ls_dl
            )
            n_eval += ev

            if flow < best_flow:
                best_flow, best_seq, best_starts = flow, seq[:], starts

        return best_seq, best_flow, best_starts, n_eval

    # ------------------------------------------------------------------

    def solve(self, initial_solution=None):
        """
        Ejecuta MS-LNS-SA.

        Returns
        -------
        job_start_times  : tiempos de inicio de cada trabajo
        flow_time        : suma de tiempos de completación
        computation_time : tiempo de cómputo en milisegundos
        n_solutions      : total de soluciones evaluadas
        """
        start_t  = time.time()
        deadline = start_t + self.time_limit

        # ── FASE 1: Multi-Start ───────────────────────────────────────
        best_seq, best_flow, best_starts, n_eval = self._multi_start(deadline)

        # Si se pasó initial_solution úsarla si es mejor
        if initial_solution is not None:
            seq0 = sorted(range(self.n), key=lambda j: initial_solution[j])
            f0, s0 = evaluate_sequence(seq0, self._algo)
            n_eval += 1
            if f0 < best_flow:
                best_flow, best_seq, best_starts = f0, seq0[:], s0

        cur_seq, cur_flow = best_seq[:], best_flow

        # ── Temperatura inicial (auto si no se especificó) ────────────
        T = self.sa_t0 if self.sa_t0 is not None else max(1.0, best_flow * 0.02)

        # ── FASE 2: Bucle LNS + SA ────────────────────────────────────
        q          = self.q_init
        no_improve = 0   # iteraciones sin mejora global

        while time.time() < deadline:

            # ── DESTRUCCIÓN ───────────────────────────────────────────
            partial, removed = _destroy(cur_seq, q, self.rng)

            # ── REPARACIÓN ────────────────────────────────────────────
            repair_dl = min(deadline, time.time() + 30.0)
            new_seq, ev = _repair(partial, removed, self._algo, repair_dl)
            n_eval += ev

            # Búsqueda local rápida post-reparación
            ls_dl = min(deadline, time.time() + 5.0)
            new_seq, new_flow, new_starts, ev2 = _quick_local_search(
                new_seq, self._algo, self.max_range, ls_dl
            )
            n_eval += ev2

            # ── CRITERIO DE ACEPTACIÓN (Simulated Annealing) ──────────
            delta = new_flow - cur_flow
            if delta < 0:
                # Mejora → aceptar siempre
                cur_seq, cur_flow = new_seq, new_flow
                if new_flow < best_flow:
                    best_flow, best_seq, best_starts = new_flow, new_seq[:], new_starts
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Empeora → aceptar con probabilidad SA
                prob = math.exp(-delta / max(T, 1e-10))
                if self.rng.random() < prob:
                    cur_seq, cur_flow = new_seq, new_flow
                no_improve += 1

            # ── Enfriar temperatura ───────────────────────────────────
            T *= self.sa_cooling

            # ── Adaptar q ────────────────────────────────────────────
            # Si llevamos muchas iteraciones sin mejorar el global,
            # aumentar q para destruir más y explorar vecindarios más lejanos
            if no_improve > 0 and no_improve % self.no_improve_q_step == 0:
                q = min(q + 1, self.q_max)
            # Si mejoró recientemente, volver al q inicial
            if no_improve == 0:
                q = self.q_init

        comp_time = (time.time() - start_t) * 1000
        return best_starts, best_flow, comp_time, n_eval
