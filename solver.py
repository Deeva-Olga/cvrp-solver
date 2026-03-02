import math, random, time, os, glob, json
import numpy as np
from typing import List, Dict, Tuple

class CVRP:
    def __init__(self, filepath: str, sol_filepath: str = None):
        self.capacity = 0
        self.coords = {}
        self.demands = {}
        self.optimal_cost = None

        with open(filepath) as f:
            lines = f.readlines()

        state = None
        for line in lines:
            line = line.strip()
            if not line or line == 'EOF': continue

            if 'CAPACITY' in line:
                self.capacity = int(line.split(':')[-1])
            elif 'NODE_COORD_SECTION' in line:
                state = 'coords'
            elif 'DEMAND_SECTION' in line:
                state = 'demands'
            elif 'DEPOT_SECTION' in line:
                break
            elif state == 'coords' and len(line.split()) >= 3:
                nid, x, y = map(float, line.split())
                self.coords[int(nid)] = (x, y)
            elif state == 'demands' and len(line.split()) >= 2:
                nid, d = map(float, line.split())
                self.demands[int(nid)] = int(d)

        self.depot = 1
        self.customers = [n for n in self.coords if n != self.depot]
        self._build_dist()

        # Загрузка оптимального решения из .sol файла
        if sol_filepath and os.path.exists(sol_filepath):
            self._load_optimal_solution(sol_filepath)

    def _load_optimal_solution(self, sol_filepath: str):
        """Загрузка оптимальной стоимости из .sol файла"""
        with open(sol_filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith('Cost'):
                    try:
                        self.optimal_cost = int(line.split()[1])
                        break
                    except:
                        pass

    def _build_dist(self):
        # Округляем расстояния до целых чисел (правила CVRPLib)
        self.dist = {}
        for i in self.coords:
            for j in self.coords:
                if i != j:
                    x1, y1 = self.coords[i]
                    x2, y2 = self.coords[j]
                    self.dist[(i, j)] = round(math.hypot(x1-x2, y1-y2))

    def route_cost(self, route: List[int]) -> Tuple[float, int]:
        cost, load = 0.0, 0
        for i in range(len(route)-1):
            cost += self.dist.get((route[i], route[i+1]), 0)
            if route[i] != self.depot:
                load += self.demands.get(route[i], 0)
        return cost, load

    def solution_cost(self, sol: List[List[int]]) -> float:
        total, served = 0.0, set()
        for r in sol:
            c, l = self.route_cost(r)
            total += c + max(0, l - self.capacity) * 10000
            served.update(n for n in r if n != self.depot)
        return total + len(set(self.customers) - served) * 100000

class HybridCVRPSolver:
    """Гибридный решатель: Clarke-Wright + интенсивный локальный поиск"""

    def __init__(self, problem: CVRP):
        self.p = problem

    def _clarke_wright(self) -> List[List[int]]:
        """Алгоритм экономии (Clarke-Wright) для начального решения"""
        routes = [[self.p.depot, c, self.p.depot] for c in self.p.customers]
        savings = []

        for i in self.p.customers:
            for j in self.p.customers:
                if i != j:
                    s = (self.p.dist[(self.p.depot, i)] +
                         self.p.dist[(self.p.depot, j)] -
                         self.p.dist.get((i, j), 0))
                    savings.append((s, i, j))

        savings.sort(reverse=True)

        for _, i, j in savings:
            route_i = next((r for r in routes if i in r), None)
            route_j = next((r for r in routes if j in r), None)

            if route_i is route_j or route_i is None or route_j is None:
                continue

            if route_i[-2] != i or route_j[1] != j:
                continue

            load_i = sum(self.p.demands[n] for n in route_i[1:-1])
            load_j = sum(self.p.demands[n] for n in route_j[1:-1])
            if load_i + load_j > self.p.capacity:
                continue

            new_route = route_i[:-1] + route_j[1:]
            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(new_route)

        return routes

    def _two_opt_intra(self, route: List[int]) -> Tuple[List[int], float]:
        """2-opt внутри маршрута"""
        best_route, best_cost = route[:], self.p.route_cost(route)[0]
        improved = True

        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = self.p.route_cost(new_route)[0]
                    if new_cost + 1e-6 < best_cost:
                        best_route, best_cost = new_route[:], new_cost
                        improved = True
            route = best_route[:]

        return best_route, best_cost

    def _two_opt_star(self, routes: List[List[int]]) -> List[List[int]]:
        """2-opt* между маршрутами (межмаршрутный обмен)"""
        best_routes, best_cost = [r[:] for r in routes], self.p.solution_cost(routes)
        n = len(routes)

        for i in range(n):
            for j in range(i+1, n):
                r1, r2 = routes[i], routes[j]
                if len(r1) < 4 or len(r2) < 4:
                    continue

                for a in range(1, len(r1)-2):
                    for b in range(1, len(r2)-2):
                        new_r1 = r1[:a+1] + r2[b+1:]
                        new_r2 = r2[:b+1] + r1[a+1:]

                        load1 = sum(self.p.demands.get(n,0) for n in new_r1[1:-1])
                        load2 = sum(self.p.demands.get(n,0) for n in new_r2[1:-1])
                        if load1 > self.p.capacity or load2 > self.p.capacity:
                            continue

                        new_routes = [r[:] for r in routes]
                        new_routes[i], new_routes[j] = new_r1, new_r2
                        new_cost = self.p.solution_cost(new_routes)

                        if new_cost + 1e-6 < best_cost:
                            best_routes, best_cost = [r[:] for r in new_routes], new_cost

        return best_routes

    def _relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """Перемещение клиента между маршрутами"""
        best_routes, best_cost = [r[:] for r in routes], self.p.solution_cost(routes)

        for i, r1 in enumerate(routes):
            if len(r1) <= 3: continue

            for pos in range(1, len(r1)-1):
                customer = r1[pos]
                new_r1 = r1[:pos] + r1[pos+1:]

                if len(new_r1) == 2:
                    candidate_routes = [r[:] for r in routes if r is not r1]
                else:
                    candidate_routes = [r[:] for r in routes]
                    candidate_routes[i] = new_r1

                for j, r2 in enumerate(candidate_routes):
                    if j == i and len(new_r1) == 2: continue

                    for ins_pos in range(1, len(r2)):
                        new_r2 = r2[:ins_pos] + [customer] + r2[ins_pos:]

                        load = sum(self.p.demands.get(n,0) for n in new_r2[1:-1])
                        if load > self.p.capacity:
                            continue

                        test_routes = [r[:] for r in candidate_routes]
                        test_routes[j] = new_r2
                        cost = self.p.solution_cost(test_routes)

                        if cost + 1e-6 < best_cost:
                            best_routes, best_cost = [r[:] for r in test_routes], cost

        return best_routes

    def solve(self, max_time=30.0) -> Dict:
        """Основной решатель с интенсивным локальным поиском"""
        start_time = time.time()

        routes = self._clarke_wright()

        for i in range(len(routes)):
            routes[i], _ = self._two_opt_intra(routes[i])

        best_routes = [r[:] for r in routes]
        best_cost = self.p.solution_cost(routes)
        no_improve = 0

        while time.time() - start_time < max_time and no_improve < 5:
            improved = False

            new_routes = self._two_opt_star(routes)
            if self.p.solution_cost(new_routes) + 1e-6 < best_cost:
                routes, best_cost = new_routes, self.p.solution_cost(new_routes)
                best_routes = [r[:] for r in routes]
                improved = True

            new_routes = self._relocate(routes)
            if self.p.solution_cost(new_routes) + 1e-6 < best_cost:
                routes, best_cost = new_routes, self.p.solution_cost(new_routes)
                best_routes = [r[:] for r in routes]
                improved = True

            for i in range(len(routes)):
                routes[i], _ = self._two_opt_intra(routes[i])

            current_cost = self.p.solution_cost(routes)
            if current_cost + 1e-6 < best_cost:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]
                improved = True

            if not improved:
                no_improve += 1
                if no_improve >= 3 and len(routes) > 1:
                    i, j = random.sample(range(len(routes)), 2)
                    if len(routes[i]) > 3 and len(routes[j]) > 3:
                        c1 = routes[i].pop(random.randint(1, len(routes[i])-2))
                        c2 = routes[j].pop(random.randint(1, len(routes[j])-2))
                        routes[i].insert(random.randint(1, len(routes[i])-1), c2)
                        routes[j].insert(random.randint(1, len(routes[j])-1), c1)
                        no_improve = 0
            else:
                no_improve = 0

        elapsed = time.time() - start_time
        return {
            'cost': best_cost,
            'routes': best_routes,
            'time': elapsed
        }

def run_experiments():
    os.makedirs('results', exist_ok=True)
    results = []

    for set_type in ['E', 'F', 'M', 'P']:
        pattern = f'data/{set_type}/*.vrp'
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"Пропускаем набор {set_type} (папка пуста)")
            continue

        print(f"\n{'='*70}")
        print(f"НАБОР {set_type}: {len(files)} задач")
        print('='*70)

        set_deviations = []

        for filepath in files:
            fname = os.path.splitext(os.path.basename(filepath))[0]
            sol_filepath = filepath.replace('.vrp', '.sol')  # Ключевое изменение!
            optimal = None

            try:
                problem = CVRP(filepath, sol_filepath)
                optimal = problem.optimal_cost

                if optimal is None:
                    print(f"{fname:15} | оптимум не найден в .sol файле, пропускаем")
                    continue

                solver = HybridCVRPSolver(problem)
                res = solver.solve(max_time=15.0)

                deviation = (res['cost'] - optimal) / optimal * 100
                set_deviations.append(deviation)

                status = "Y" if deviation <= 10 else "W" if deviation <= 15 else "N"
                print(f"{status} {fname:15} | найдено: {res['cost']:7.1f} | оптимум: {optimal:4d} | "
                      f"откл: {deviation:5.1f}% | время: {res['time']:4.1f}s | маршр: {len(res['routes'])}")

                results.append({
                    'file': fname,
                    'set': set_type,
                    'optimal': optimal,
                    'found': res['cost'],
                    'deviation': deviation,
                    'time': res['time'],
                    'routes': len(res['routes'])
                })
            except Exception as e:
                print(f"{fname:15} | ОШИБКА: {str(e)[:50]}")

        if set_deviations:
            avg_dev = sum(set_deviations) / len(set_deviations)
            grade = "8-10" if avg_dev <= 10 else "6-7" if avg_dev <= 15 else "5" if avg_dev <= 20 else "4" if avg_dev <= 25 else "3"
            print(f"\nИтоги набора {set_type}: среднее отклонение = {avg_dev:5.2f}%  оценка {grade}")

    if results:
        all_deviations = [r['deviation'] for r in results if r['deviation'] is not None]
        overall_avg = sum(all_deviations) / len(all_deviations)
        grade = "8-10" if overall_avg <= 10 else "6-7" if overall_avg <= 15 else "5" if overall_avg <= 20 else "4" if overall_avg <= 25 else "3"

        print("\n" + "="*70)
        print(f"ОБЩИЙ РЕЗУЛЬТАТ: среднее отклонение = {overall_avg:5.2f}%  ИТОГОВАЯ ОЦЕНКА: {grade}")
        print("="*70)

        with open('results/cvrp_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_instances': len(results),
                    'average_deviation': overall_avg,
                    'grade': grade
                },
                'details': results
            }, f, indent=2)
        print("\nРезультаты сохранены в 'results/cvrp_results.json'")

    return results

if __name__ == "__main__":

    print("РЕШЕНИЕ CVRP: ГИБРИДНЫЙ АЛГОРИТМ (Clarke-Wright + Локальный поиск)")
    print("Валидация через официальные .sol файлы (округление расстояний до целых)")
    print("\n")

    sets = ['E', 'F', 'M', 'P']
    missing_sets = []
    for s in sets:
        vrp_files = glob.glob(f'data/{s}/*.vrp')
        sol_files = glob.glob(f'data/{s}/*.sol')
        if not vrp_files:
            missing_sets.append(s)
        else:
            print(f"Набор {s}: {len(vrp_files)} задач, {len(sol_files)} решений")

    if missing_sets:
        print(f"\n Отсутствуют файлы для наборов: {', '.join(missing_sets)}")
        print("Убедитесь, что структура папок:")
        print("  data/E/  E-*.vrp + E-*.sol")
        print("  data/F/  F-*.vrp + F-*.sol")
        print("  ...и т.д.")

    #input("\nНажмите Enter для запуска экспериментов...")
    run_experiments()