"""
Módulo para leer instancias del problema NWJSSP
Formato: Primera línea con n (trabajos) y m (máquinas)
Siguientes n líneas con datos de operaciones y tiempo de liberación
"""

def read_nwjssp_instance(filename):
    """
    Lee una instancia del problema NWJSSP
    
    Args:
        filename: Ruta del archivo de instancia
        
    Returns:
        n: número de trabajos
        m: número de máquinas
        operations: lista de operaciones por trabajo
        release_dates: tiempos de liberación de cada trabajo
        L: lista de maquinas y tiempos de procesamiento
    """
    with open(filename, 'r') as f:
        # Primera línea: número de trabajos y máquinas
        first_line = f.readline().strip().split()
        n = int(first_line[0])
        m = int(first_line[1])
        
        operations = []
        release_dates = []
        L = []
        
        # Leer datos de cada trabajo
        for i in range(n):
            line = list(map(int, f.readline().strip().split()))
            
            # Los últimos valores son: máquina, tiempo, máquina, tiempo, ..., release_date
            release_date = line[-1]
            release_dates.append(release_date)
            
            # Extraer operaciones (pares máquina-tiempo)
            job_ops = []
            for j in range(0, len(line) - 1, 2):
                machine = line[j]
                processing_time = line[j + 1]
                job_ops.append((machine, processing_time))
            
            operations.append(job_ops)
            L.append(job_ops)
        
        return n, m, operations, release_dates, L


def calculate_flow_time(job_start_times, operations, release_dates):
    """
    Calcula el tiempo de flujo total de una solución
    
    Args:
        job_start_times: lista con tiempos de inicio de cada trabajo
        operations: lista de operaciones por trabajo
        release_dates: tiempos de liberación
        
    Returns:
        flow_time: tiempo de flujo total
        completion_times: tiempos de finalización de cada trabajo
    """
    n = len(job_start_times)
    completion_times = []
    
    for i in range(n):
        current_time = max(job_start_times[i], release_dates[i])
        
        # Procesar cada operación del trabajo
        for machine, processing_time in operations[i]:
            current_time += processing_time
        
        completion_times.append(current_time)
    
    flow_time = sum(completion_times)
    return flow_time, completion_times


def is_feasible_solution(job_start_times, operations, release_dates):
    """
    Verifica si una solución es factible verificando:
    - Cada trabajo respeta su fecha de liberación
    - No hay conflictos de máquinas
    
    Args:
        job_start_times: lista con tiempos de inicio de cada trabajo
        operations: lista de operaciones por trabajo
        release_dates: tiempos de liberación
        
    Returns:
        feasible: booleano indicando si la solución es factible
    """
    n = len(job_start_times)
    
    # Crear tabla de eventos por máquina
    machine_schedule = {}
    
    for i in range(n):
        current_time = max(job_start_times[i], release_dates[i])
        
        for op_idx, (machine, processing_time) in enumerate(operations[i]):
            if machine not in machine_schedule:
                machine_schedule[machine] = []
            
            start = current_time
            end = start + processing_time
            
            # Verificar conflictos con otras operaciones en la misma máquina
            for prev_start, prev_end in machine_schedule[machine]:
                if (start < prev_end and end > prev_start):
                    return False
            
            machine_schedule[machine].append((start, end))
            current_time = end
    
    return True


def get_schedule_from_solution(job_start_times, operations):
    """
    Obtiene el programa de máquinas a partir de los tiempos de inicio de trabajos
    
    Args:
        job_start_times: lista con tiempos de inicio de cada trabajo
        operations: lista de operaciones por trabajo
        
    Returns:
        schedule: diccionario con operaciones por máquina y sus tiempos
    """
    schedule = {}
    n = len(job_start_times)
    
    for i in range(n):
        current_time = job_start_times[i]
        
        for machine, processing_time in operations[i]:
            if machine not in schedule:
                schedule[machine] = []
            
            schedule[machine].append({
                'job': i,
                'start': current_time,
                'end': current_time + processing_time
            })
            current_time += processing_time
    
    return schedule
