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

---
Optimizaciones de velocidad respecto a la versión original:

1. _repair con VENTANA RESTRINGIDA (repair_window):
   En vez de evaluar las n+1 posiciones de inserción para cada trabajo
   eliminado, se evalúa solo una ventana de `repair_window` posiciones
   centrada donde el trabajo "encajaría" según su release_date relativa.
   Costo: O(q × w × n)  en vez de  O(q × n × n).

2. PRESUPUESTO DINÁMICO POR ITERACIÓN (iter_time_frac):
   El tiempo asignado a cada iteración LNS se calcula como una fracción
   del tiempo restante total, no como un valor fijo (el original usaba
   30 s + 5 s fijos, lo que limitaba el bucle a ~100 iteraciones en 1 h).
   Con iter_time_frac=0.05 y mucho tiempo restante el presupuesto puede
   ser alto; a medida que se acerca el deadline se reduce automáticamente.

3. BÚSQUEDA LOCAL POST-REPARACIÓN LIVIANA (full=False):
   Después de reparar solo se aplica N1 (2-opt consecutivo), que es O(n).
   La búsqueda completa N1+N2+N3 se reserva para el multi-start inicial,
   donde vale la pena invertir más tiempo en cada solución candidata.
"""

import time
import random
import math
from constructive import ConstructiveAlgorithm
from vnd import evaluate_sequence, _two_opt_FI, _swap_FI, _insertion_FI


# ---------------------------------------------------------------------------
# Búsqueda local rápida adaptativa
# ---------------------------------------------------------------------------

def _quick_local_search(seq, algo, max_range, deadline, full=True):
    """
    Una pasada de vecindarios FI.

    full=True  → N1 + N2 + N3  (para multi-start, donde conviene explorar bien)
    full=False → solo N1 (2-opt consecutivo), O(n), para post-reparación rápida.

    Retorna (secuencia, flow, starts, n_soluciones_evaluadas).
    """
    n_eval = 0
    neighbors = [(_two_opt_FI, False)]
    if full:
        neighbors += [(_swap_FI, True), (_insertion_FI, True)]

    for fn, needs_range in neighbors:
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
    Estrategia mixta: mitad aleatoria, mitad los más 'tardíos' (últimas posiciones).
    Retorna (secuencia_parcial, lista_de_trabajos_eliminados).
    """
    n = len(seq)
    q = max(2, min(q, n - 1))

    n_random = max(1, q // 2)
    n_tail   = q - n_random

    tail_candidates = seq[max(0, n - q * 2):]
    tail_jobs = rng.sample(tail_candidates, min(n_tail, len(tail_candidates)))
    tail_set  = set(tail_jobs)

    remaining_pool = [j for j in seq if j not in tail_set]
    random_jobs = rng.sample(remaining_pool, min(n_random, len(remaining_pool)))

    removed = list(tail_set) + random_jobs
    partial  = [j for j in seq if j not in set(removed)]
    return partial, removed


# ---------------------------------------------------------------------------
# Operador de REPARACIÓN greedy con ventana restringida
# ---------------------------------------------------------------------------

def _repair(partial_seq, removed_jobs, algo, deadline, repair_window=None):
    """
    Reinserta los trabajos eliminados en la mejor posición dentro de una
    ventana de tamaño `repair_window` alrededor de la posición estimada
    según la release_date relativa del trabajo.

    repair_window=None → evalúa todas las posiciones (original, lento).
    repair_window=20   → evalúa ≤20 posiciones por trabajo (recomendado).

    La posición central de la ventana se estima proporcionalmente a la
    release_date del trabajo dentro del rango de release_dates de la
    secuencia parcial, lo que suele colocar el trabajo cerca de su lugar
    óptimo sin necesidad de explorar toda la secuencia.

    Costo: O(q × repair_window × n) vs O(q × n²) del original.

    Retorna (secuencia_completa, n_soluciones_evaluadas).
    """
    seq = partial_seq[:]
    n_eval = 0

    for job in removed_jobs:
        if time.time() >= deadline:
            seq.append(job)
            continue

        current_len = len(seq)

        if repair_window is None or repair_window >= current_len + 1:
            # Sin restricción: evaluar todas las posiciones (comportamiento original)
            pos_range = range(current_len + 1)
        else:
            # Estimar posición proporcional a la release_date del trabajo
            rd = algo.release_dates[job]
            rds = [algo.release_dates[j] for j in seq]
            min_rd, max_rd = min(rds), max(rds)
            if max_rd > min_rd:
                center = int((rd - min_rd) / (max_rd - min_rd) * current_len)
            else:
                center = current_len // 2
            center = max(0, min(center, current_len))

            half = repair_window // 2
            lo = max(0, center - half)
            hi = min(current_len + 1, lo + repair_window)
            lo = max(0, hi - repair_window)   # corregir si hi chocó con el límite
            pos_range = range(lo, hi)

        best_flow = float('inf')
        best_pos  = current_len   # fallback: insertar al final

        for pos in pos_range:
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

    Parámetros originales
    ---------------------
    max_range          : rango para búsqueda local interna            (default 10)
    time_limit         : segundos totales                             (default 3600)
    n_starts           : soluciones iniciales del Multi-Start         (default 5)
    q_init             : trabajos eliminados en destrucción inicial    (default 4)
    q_max              : máximo trabajos a eliminar                   (default 10)
    no_improve_q_step  : iteraciones sin mejora antes de subir q      (default 15)
    sa_t0              : temperatura inicial del SA  (None → auto)    (default None)
    sa_cooling         : factor de enfriamiento geométrico            (default 0.995)
    seed               : semilla aleatoria                            (default 42)

    Parámetros nuevos (control de velocidad)
    ----------------------------------------
    repair_window  : número de posiciones evaluadas en _repair por trabajo.
                     Reducir para instancias grandes (ej: 10–15).
                     None → evalúa todas (original, lento).           (default 20)
    iter_time_frac : fracción del tiempo restante asignada a cada
                     iteración LNS completa. Valores más bajos permiten
                     más iteraciones totales.                          (default 0.05)
    max_no_improve : iteraciones globales sin mejorar el mejor antes
                     de declarar convergencia y parar anticipadamente.
                     None → nunca para antes del time_limit.          (default 200)
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
                 seed: int = 42,
                 repair_window: int = 20,
                 iter_time_frac: float = 0.05,
                 max_no_improve: int = 200):
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
        self.repair_window = repair_window
        self.iter_time_frac = iter_time_frac
        self.max_no_improve = max_no_improve
        self._algo = ConstructiveAlgorithm(n, m, operations, release_dates)

    # ------------------------------------------------------------------

    def _multi_start(self, deadline):
        """
        Genera n_starts soluciones diversas con búsqueda local completa (N1+N2+N3)
        y devuelve la mejor como punto de partida del bucle LNS.
        """
        best_seq, best_flow, best_starts = None, float('inf'), None
        n_eval = 0

        # Presupuesto por solución: máximo 5 s, o tiempo restante / starts
        time_remaining = deadline - time.time()
        budget = min(5.0, time_remaining / (self.n_starts + 1))

        for _ in range(self.n_starts):
            if time.time() >= deadline:
                break

            sol0, _ = _build_random_solution(self._algo, self.rng)
            seq = sorted(range(self.n), key=lambda j: sol0[j])

            ls_dl = min(deadline, time.time() + budget)
            seq, flow, starts, ev = _quick_local_search(
                seq, self._algo, self.max_range, ls_dl, full=True
            )
            n_eval += ev

            if flow < best_flow:
                best_flow, best_seq, best_starts = flow, seq[:], starts

        return best_seq, best_flow, best_starts, n_eval

    # ------------------------------------------------------------------

    def _iter_budget(self, deadline):
        """
        Presupuesto de tiempo (segundos) para la iteración LNS actual.

        Se calcula como una fracción del tiempo restante, acotada a 60 s.
        Esto garantiza que el presupuesto se reduce automáticamente a medida
        que se acerca el deadline, permitiendo siempre más iteraciones en vez
        de bloquear el bucle con valores fijos grandes.
        """
        remaining = max(0.0, deadline - time.time())
        return min(60.0, remaining * self.iter_time_frac)

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

        # Incorporar solución inicial externa si es mejor
        if initial_solution is not None:
            seq0 = sorted(range(self.n), key=lambda j: initial_solution[j])
            f0, s0 = evaluate_sequence(seq0, self._algo)
            n_eval += 1
            if f0 < best_flow:
                best_flow, best_seq, best_starts = f0, seq0[:], s0

        cur_seq, cur_flow = best_seq[:], best_flow

        # ── Temperatura inicial (auto) ────────────────────────────────
        T = self.sa_t0 if self.sa_t0 is not None else max(1.0, best_flow * 0.02)

        # ── FASE 2: Bucle LNS + SA ────────────────────────────────────
        q          = self.q_init
        no_improve = 0

        while time.time() < deadline:

            # ── PARADA POR CONVERGENCIA ───────────────────────────────
            if self.max_no_improve is not None and no_improve >= self.max_no_improve:
                break

            # Presupuesto dinámico para esta iteración
            budget       = self._iter_budget(deadline)
            iter_start   = time.time()
            iter_deadline = min(deadline, iter_start + budget)

            # ── DESTRUCCIÓN ───────────────────────────────────────────
            partial, removed = _destroy(cur_seq, q, self.rng)

            # ── REPARACIÓN con ventana restringida ────────────────────
            # Asignar 60% del presupuesto de la iteración a la reparación
            repair_dl = min(iter_deadline, iter_start + budget * 0.6)
            new_seq, ev = _repair(
                partial, removed, self._algo, repair_dl,
                repair_window=self.repair_window
            )
            n_eval += ev

            # ── Búsqueda local post-reparación (solo N1, liviana) ─────
            # Asignar 35% del presupuesto a la búsqueda local
            ls_dl = min(iter_deadline, time.time() + budget * 0.35)
            new_seq, new_flow, new_starts, ev2 = _quick_local_search(
                new_seq, self._algo, self.max_range, ls_dl, full=False
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
            if no_improve > 0 and no_improve % self.no_improve_q_step == 0:
                q = min(q + 1, self.q_max)
            if no_improve == 0:
                q = self.q_init

        comp_time = (time.time() - start_t) * 1000
        return best_starts, best_flow, comp_time, n_eval