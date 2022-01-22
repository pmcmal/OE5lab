from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based import HGS
from mealpy.problem import Problem
from mealpy.utils.termination import Termination

# Program dzieli się na sekcje od A do E:

# USTAWIENIE PARAMETRÓW
# A - Inny sposób podawania dolnej i górnej granicy. Oto kilka przykładów:
## A1. Kiedy masz różne dolne i górne granice dla każdego parametru
problem_dict1 = {
    "obj_func": F5,
    "lb": [-3, -5, 1, -10, ],
    "ub": [5, 10, 100, 30, ],
    "minmax": "min",
    "verbose": True,
}
problem_obj1 = Problem(problem_dict1)

## A2. Jeśli masz taką samą dolną i górną granicę dla każdego parametru, wtedy możesz użyć:
##      + int lub float: następnie musisz określić rozmiar problemu / liczbę wymiarów (n_dims)
problem_dict2 = {
    "obj_func": F5,
    "lb": -10,
    "ub": 30,
    "minmax": "min",
    "verbose": True,
    "n_dims": 30,  # Pamiętaj o "n_dims"
}
problem_obj2 = Problem(problem_dict2)

##      + array: 2 sposoby
problem_dict3 = {
    "obj_func": F5,
    "lb": [-5],
    "ub": [10],
    "minmax": "min",
    "verbose": True,
    "n_dims": 30,  # Pamiętaj o "n_dims"
}
problem_obj3 = Problem(problem_dict3)

n_dims = 100
problem_dict4 = {
    "obj_func": F5,
    "lb": [-5] * n_dims,
    "ub": [10] * n_dims,
    "minmax": "min",
    "verbose": True,
}

## B - Istnieją 4 przypadki zakończenia:
### 1. FE (Liczba ocen funkcji)
### 2. MG (Maximum Generacji / Epok): Domyślna wartość
### 3. ES (Wczesne zatrzymanie): Same idea in training neural network (Jeśli najlepsze rozwiązanie globalne nie jest lepsze niż epsilon
###     po K epokach, następnie zatrzymać program)
### 4. TB (Granice czasowe): Chcesz aby algorytm działał K sekund. Np przy porównaniu różnych algorytmów

termination_dict1 = {
    "mode": "FE",
    "quantity": 100000  # liczba ocen funkcji
}
termination_dict2 = {  # Podczas tworzenia obiektu, będzie on nadpisywał domyślną epokę zdefiniowaną w modelu
    "mode": "MG",
    "quantity": 1000  # liczba epok
}
termination_dict3 = {
    "mode": "ES",
    "quantity": 20  # po ilu epokach jeśli wynik nie ulegnie poprawy zatrzymany zostanie program
}
termination_dict4 = {
    "mode": "ES",
    "quantity": 60  # Czas uruchomienia (przypadek 4)
}
# wybór
termination_obj1 = Termination(termination_dict3)


### Przekaż obiekt zakończenia do modelu jako dodatkowy parametr pod słowem kluczowym termination
# model3 = HGS.OriginalHGS(problem_dict1, epoch=100, pop_size=50, PUP=0.08, LH=10000, termination=termination_obj1)
# model3.solve()
### Nie można przekazać obiektu termination_dict! a jedynie termination.

# C - Test z różnymi trybami szkolenia (sekwencyjny, paralelizacja wątków, paralelizacja przetwarzania)
## + sequential: (sekwencyjny) domyślny (jeden rdzeń)
## + thread: wiele wątków w zależności od używanego CPU
## + process: wiele rdzeni do uruchomienia algorytmu

model1 = HGS.OriginalHGS(problem_dict2, epoch=100, pop_size=50, PUP=0.08, LH=10000)
model1.solve(mode='sequential')

# model1 = HGS.OriginalHGS(problem_dict1, epoch=100, pop_size=50, PUP=0.08, LH=10000)
# model1.solve(mode='thread')
# process w windows nalezy uzyc if __name__:
 # if __name__ == "__main__":
   # model1 = HGS.OriginalHGS(problem_dict1, epoch=100, pop_size=50, PUP=0.08, LH=10000)
  #  model1.solve(mode='process')

# D - Wykresy wszystkich dostępnych danych
## Istnieje 8 różnych wykresów:
## D.1: Wartość fitness:
##      1. Global best fitness
##      2. Local best fitness
## D.2: Wartośc obiektywna:
##      3. Global objective
##      4. Local objective
## D.3: Czas pracy (dla każdej epoki)
##      5. Runtime
## D.4: Eksploracja a eksploatacja
##      6. Exploration vs Exploitation
## D.5: Różnorodność populacji
##      7. Diversity
## D.6: Wartość trajektorii (tylko 1D i 2D!)
##      8. Trajectory

## model8 = HGS.OriginalHGS(problem_dict1, epoch=100, pop_size=50, PUP=0.08, LH=10000)
## model8.solve()

## Dostęp do każdego można uzyskać obiektem "history":
# model1.history.save_global_objectives_chart(filename="HGS/goc")
# model1.history.save_local_objectives_chart(filename="HGS/loc")
model1.history.save_global_best_fitness_chart(filename="HGS/global_best_fitness")
# model1.history.save_local_best_fitness_chart(filename="HGS/lbfc")
model1.history.save_runtime_chart(filename="HGS/runtime")
model1.history.save_exploration_exploitation_chart(filename="HGS/exploration_exploitation")
# model1.history.save_diversity_chart(filename="HGS/dc")
# model1.history.save_trajectory_chart(list_agent_idx=[3, 5], list_dimensions=[3], filename="HGS/tc")

# E - obsługa funkcji wieloobiektowej i metody ograniczeń
## Do obsługi wielu celów mealpy używa metody ważenia, przekształca wiele celów w jeden cel (wartość fitness)
## Zdefiniowanie funkcji celu, ograniczeń
def obj_function(solution):
    f1 = solution[0] ** 2
    f2 = ((2 * solution[1]) / 5) ** 2
    f3 = 0
    for i in range(3, len(solution)):
        f3 += (1 + solution[i] ** 2) ** 0.5
    return [f1, f2, f3]

## Zdefiniuj wagi:
### f1=50%,f2=20%,f3=30% -> [0.5, 0.2, 0.3] -> wartość fitness = 0.5*f1 + 0.2*f2 + 0.3*f3
### Domyślna waga to [1, 1, 1]
problem_dict1 = {
    "obj_func": obj_function,
    "lb": [-3, -5, 1, -10, ],
    "ub": [5, 10, 100, 30, ],
    "minmax": "min",
    "verbose": True,
    "obj_weight": [0.5, 0.2, 0.1]  # pamiętaj o "obj_weight"
}
# problem_obj9 = Problem(problem_dict9)
# model9 = HGS.OriginalHGS(problem_obj9, epoch=100, pop_size=50, PUP=0.08, LH=10000)
# model9.solve()

## Aby uzyskać dostęp do wyników, można je uzyskać za pomocą metody solve()
# position, fitness_value = model9.solve()

## Uzyskanie wartości fitness i wartości obiektywnych za pomocą atrybutu "solution"
## A agent / solution format [position, [fitness, [obj1, obj2, ..., obj_n]]]
# position = model9.solution[0]
# fitness_value = model9.solution[1][0]
# objective_values = model9.solution[1][1]