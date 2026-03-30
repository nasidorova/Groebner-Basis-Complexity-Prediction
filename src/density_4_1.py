from memory_profiler import memory_usage
from ginv.monom import Monom
from ginv.poly import Poly
from ginv.gb import GB
from ginv.ginv import *
import json
import os
import math
import re
import pandas as pd
from typing import List, Callable

x1 = x2 = x3 = x4 = x5 = None
a = b = c = None
u0 = u1 = u2 = u3 = u4 = u5 = u6 = None

VERY_QUICK = ['quadfor2', 'sparse5', 'hunecke', 'solotarev', 'chandra4', 'quadgrid', 'lorentz', 'liu', 'hemmecke', 'boon', 'chandra5', 'caprasse', 'issac97', 'hcyclic5', 'redcyc5', 'cyclic5', 'extcyc4', 'chemequ', 'uteshev_bikker', 'chandra6', 'geneig']
QUICK = ['chemequs', 'vermeer', 'camera1s', 'reimer4', 'redeco7', 'tangents', 'cassou', 'butcher', 'eco7', 'cohn2', 'dessin1', 'des18_3', 'hcyclic6', 'noon5', 'katsura6', 'cyclic6', 'butcher8', 'redcyc6', 'cpdm5', 'extcyc5']
MEDIUM = ['noon6', 'reimer5', 'kotsireas', 'assur44']
ADD = ['s9_1', 'rose', 'speer']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GINDIST_DIR = os.path.join(CURRENT_DIR, 'GInvDist')
JSON_DIR = os.path.join(GINDIST_DIR, 'json')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'Results')


def total_degree_of_monom(monom_str, variables):
    monom_str = monom_str.replace('^', '**')
    if monom_str.strip() == "1":
        return 0
    total = 0
    for var in variables:
        pattern = rf'{re.escape(var)}(?:\*\*|\*?)(\d*)'
        matches = re.findall(pattern, monom_str)
        for exp in matches:
            total += 1 if exp == '' else int(exp)
    return total


def split_into_monomials(eq_str):
    eq = eq_str.replace(' ', '').replace('^', '**')
    eq = eq.replace('-', '+-')
    if eq.startswith('+'):
        eq = eq[1:]
    terms = [t.strip() for t in eq.split('+') if t.strip() and t.strip() not in ('0', '-0')]
    return terms


def compute_density_and_max_degree(equations, variables):
    total_monomials = 0
    global_max_degree = 0
    for eq in equations:
        left_part = eq.split('=')[0]
        monomials = split_into_monomials(left_part)
        total_monomials += len(monomials)
        for monom in monomials:
            deg = total_degree_of_monom(monom, variables)
            if deg > global_max_degree:
                global_max_degree = deg

    n = len(variables)
    d = global_max_degree
    possible = 1 if d == 0 else math.comb(n + d, d)
    density = total_monomials / possible if possible > 0 else 0.0
    return density, total_monomials, global_max_degree


def estimate_d_reg_asymp(n, m):
    if m <= n:
        return 0
    alpha = m / n
    if alpha > 1:
        return n * (0.5 + math.sqrt(alpha * (alpha - 1)))
    return 0


def estimate_first_fall_degree(n, m):
    return math.ceil((n + m) / 2)


def estimate_non_mult_prolongations_initial(equations, variables):
    u_list = []
    for eq in equations:
        left_part = eq.split('=')[0]
        monomials = split_into_monomials(left_part)
        if monomials:
            max_monom = max(monomials, key=lambda m: total_degree_of_monom(m, variables))
            u_list.append(max_monom)

    max_degs = {var: 0 for var in variables}
    for u in u_list:
        for var in variables:
            pattern = rf'{re.escape(var)}(?:\*\*|\*?)(\d*)'
            matches = re.findall(pattern, u)
            for exp in matches:
                deg = 1 if exp == '' else int(exp)
                max_degs[var] = max(max_degs[var], deg)

    total_nm = 0
    for u in u_list:
        u_degs = {var: 0 for var in variables}
        for var in variables:
            pattern = rf'{re.escape(var)}(?:\*\*|\*?)(\d*)'
            matches = re.findall(pattern, u)
            for exp in matches:
                deg = 1 if exp == '' else int(exp)
                u_degs[var] += deg
        nm_count = sum(1 for var in variables if u_degs[var] < max_degs[var])
        total_nm += nm_count
    return total_nm


def compute_macaulay_bound(equations, variables):
    degrees = []
    for eq in equations:
        left_part = eq.split('=')[0]
        monomials = split_into_monomials(left_part)
        if monomials:
            max_d = max(total_degree_of_monom(m, variables) for m in monomials)
            degrees.append(max_d)
    return 1 + sum(d - 1 for d in degrees) if degrees else 0


def init(variables: List[str], order: Callable = Monom.TOPdeglex) -> None:
    Monom.init(variables)
    Monom.variables = variables.copy()
    Monom.zero = Monom(0 for _ in Monom.variables)
    Monom.cmp = order
    Poly.cmp = order
    for i in range(len(Monom.variables)):
        p = Poly()
        p.append([Monom(0 if l != i else 1 for l in range(len(Monom.variables))), 1])
        globals()[Monom.variables[i]] = p


def receiving_json(test_name, json_data, out=False):
    try:
        size = json_data["dimension"]
        if out:
            print(f"Тест для {test_name}")

        variables = json_data["variables"]
        init(variables)
        equations = json_data['equations']

        # Вычисляем метрики до запуска алгоритма
        density, total_monomials, max_degree_input = compute_density_and_max_degree(equations, variables)

        n = size
        m = len(equations)
        d_reg_asymp = estimate_d_reg_asymp(n, m)
        first_fall = estimate_first_fall_degree(n, m)
        non_mult_initial = estimate_non_mult_prolongations_initial(equations, variables)
        macaulay_bound = compute_macaulay_bound(equations, variables)

        data = {
            'dimension': size,
            'density': density,
            'total_monomials': total_monomials,
            'max_degree_input': max_degree_input,
            'd_reg_asymp': d_reg_asymp,
            'first_fall_estimate': first_fall,
            'non_mult_prolongations_initial': non_mult_initial,
            'macaulay_bound': macaulay_bound,
        }

        if out:
            print(f"density = {density:.6f}, max_degree_input = {max_degree_input}")

        return data

    except Exception as e:
        print(f"Error processing JSON {test_name}: {e}")
        return -1


def test_with_memory(test):
    print(test, 'start...')
    res = []
    def f():
        filepath = os.path.join(JSON_DIR, test + '.json')
        with open(filepath, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        res.append(receiving_json(test, data, True))
        return res[-1]

    mem_usage = memory_usage(f)
    res[-1]['avr memory'] = sum(mem_usage) / len(mem_usage)
    res[-1]['max memory'] = max(mem_usage)
    print(res[-1])

    result_path = os.path.join(RESULTS_DIR, test + '.json')
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(res[-1], file, ensure_ascii=False, indent=2)
    print(test, 'complete!')


def test_json():
    tests = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    print("Найдено файлов:", len(tests))
    res = []
    for test in tests:
        test_name = test.split('.')[0]
        filepath = os.path.join(JSON_DIR, test)
        if os.path.getsize(filepath) == 0:
            print(f"Skipping empty file: {test}")
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f_in:
                json_data = json.load(f_in)
            data = receiving_json(test_name, json_data)
            res.append({'name': test, **data})
        except json.JSONDecodeError as e:
            print(f"JSON error in {test}: {e}")
        except Exception as e:
            print(f"Other error in {test}: {e}")

    df = pd.DataFrame(res)
    df.to_csv('metrics_only_1.csv', sep=';', index=False)
    print('Метрики сохранены в metrics_only_1.csv ')


if __name__ == '__main__':
    print("Файлы в json:", os.listdir(JSON_DIR))
    test_json()