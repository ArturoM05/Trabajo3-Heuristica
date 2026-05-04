"""
Algoritmo Constructivo Determinístico para NWJSSP
Utiliza una estrategia greedy basada en tiempos de procesamiento
y fechas de liberación para construir una solución.
"""

import time
from read_instances import calculate_flow_time


class ConstructiveAlgorithm:
    def __init__(self, n, m, operations, release_dates):
        self.n = n
        self.m = m
        self.operations = operations
        self.release_dates = release_dates

    def calculate_priority(self, job_id):
        total_time = sum(pt for _, pt in self.operations[job_id])
        return (self.release_dates[job_id], total_time)

    def find_earliest_start_time(self, job_id, machine_schedule):
        return self._calculate_valid_start_time(job_id, machine_schedule)

    def _calculate_valid_start_time(self, job_id, machine_schedule):
        """
        Calcula analíticamente el tiempo de inicio más temprano válido.
        Genera candidatos relevantes y los evalúa en orden.
        """
        ops = self.operations[job_id]
        rd = self.release_dates[job_id]
        machines_needed = {machine for machine, _ in ops}

        # Generar candidatos: release date + todos los tiempos de liberación de máquinas necesarias
        candidates = {rd}
        for machine in machines_needed:
            if machine in machine_schedule:
                for _, end, _ in machine_schedule[machine]:
                    if end >= rd:
                        candidates.add(end)

        # Evaluar candidatos en orden ascendente
        for trial_start in sorted(candidates):
            if self._is_valid_start_time(job_id, trial_start, machine_schedule):
                return trial_start

        # Si no hay candidatos válidos (raro), retornar el máximo candidato
        # Esto evita el fallback costoso que había antes
        return max(candidates) if candidates else rd

    def _is_valid_start_time(self, job_id, start_time, machine_schedule):
        if start_time < self.release_dates[job_id]:
            return False

        current_time = start_time
        for machine, processing_time in self.operations[job_id]:
            op_start = current_time
            op_end = current_time + processing_time
            if machine in machine_schedule:
                for prev_start, prev_end, _ in machine_schedule[machine]:
                    if op_start < prev_end and op_end > prev_start:
                        return False
            current_time = op_end

        return True

    def solve(self):
        start_computation = time.time()

        job_order = sorted(range(self.n), key=self.calculate_priority)

        job_start_times = [0] * self.n
        machine_schedule = {}

        for job_id in job_order:
            start_time = self.find_earliest_start_time(job_id, machine_schedule)
            job_start_times[job_id] = start_time

            current_time = start_time
            for machine, processing_time in self.operations[job_id]:
                if machine not in machine_schedule:
                    machine_schedule[machine] = []
                machine_schedule[machine].append(
                    (current_time, current_time + processing_time, job_id)
                )
                current_time += processing_time

        flow_time, _ = calculate_flow_time(
            job_start_times, self.operations, self.release_dates
        )

        end_computation = time.time()
        computation_time = (end_computation - start_computation) * 1000

        return job_start_times, flow_time, computation_time