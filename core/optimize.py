import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

try:
    from .modeling import LLMEnergy
    # from .modeling_parition import LLMEnergy
except ImportError:
    from modeling import LLMEnergy
    # from modeling_parition import LLMEnergy


class GeneticAlgorithMulti:
    """Simple real-coded genetic algorithm used for deployment optimization."""

    def __init__(self, model, n_pop=10, epoch=100, p_muta=0.3):
        self.model = model
        self.n_pop = n_pop
        self.epoch = epoch
        self.p_muta = p_muta

    def init_population(self):
        """Initialize the population with random chromosomes."""
        self.p_greedy = self.model.up_limit_train.sum(axis=0) / self.model.up_limit_train.sum()

        pops = []
        for _ in range(self.n_pop):
            # The original code also kept an optional greedy warm start.
            pops.append(self.model(np.random.rand(self.model.dim)))
        return pops

    def select(self, pops):
        """Select parents with inverse-fitness roulette-wheel sampling."""
        n_select = len(pops) // 2 * 2

        fitness = np.array([x.obj for x in pops])
        fitness_inverse = fitness.sum() / (fitness + 1e-6)
        prob = fitness_inverse / fitness_inverse.sum()
        cumulative_prob = np.cumsum(prob)

        parents, fit_index, new_index = [], 0, 0
        random_points = sorted([random.random() for _ in range(n_select)])
        while new_index < n_select:
            if random_points[new_index] <= cumulative_prob[fit_index]:
                parents.append(pops[fit_index])
                new_index += 1
            else:
                fit_index += 1
        random.shuffle(parents)
        return parents

    def crossover_mutation(self, parents):
        """Generate offspring with arithmetic crossover and bounded mutation."""
        father, mother = parents[::2], parents[1::2]
        dim = len(father[0].chromosome)

        offspring = []
        for fa, ma in zip(father, mother):
            ch1, ch2 = deepcopy(fa.chromosome), deepcopy(ma.chromosome)
            p = np.random.rand()
            s1 = p * (ch2 - ch1) + ch1
            s2 = p * (ch1 - ch2) + ch2

            if np.random.rand() < self.p_muta:
                ind = np.random.choice(dim, int(dim * self.p_muta))
                s1[ind] = s1[ind] + np.random.rand(len(ind)) * 2 - 1
            if np.random.rand() < self.p_muta:
                ind = np.random.choice(dim, int(dim * self.p_muta))
                s2[ind] = np.random.rand(len(ind)) + np.random.rand(len(ind)) * 2 - 1

            s1, s2 = np.clip(s1, 0, 1), np.clip(s2, 0, 1)
            offspring.extend([self.model(s1), self.model(s2)])

        return offspring

    def eliminate(self, pops, reverse=False):
        """Remove duplicates and keep the best-ranked individuals."""
        _, index = np.unique(np.stack([pi.chromosome for pi in pops]), axis=0, return_index=True)
        pops = [pops[i] for i in index]
        pops = sorted(pops, key=lambda x: x.obj, reverse=reverse)
        return pops[:self.n_pop]

    def __call__(self):
        """Run the genetic algorithm and print the best solution found."""
        pops = self.init_population()
        for _ in tqdm(range(self.epoch)):
            parents = self.select(pops)
            offspring = self.crossover_mutation(parents)
            pops = self.eliminate(pops + offspring)

        print(self.model.countries[:12])
        print(self.model.countries[12:])
        for idx in range(1):
            print(
                "water:{:.5f}  carbon:{:.5f}".format(
                    pops[idx].imp_percent["water"], pops[idx].imp_percent["carbon"]
                )
            )
            print("Country-level inference shares:")
            print(("{:.4f} " * 12).format(*pops[idx].chromosome[:12]))
            print(("{:.4f} " * 12).format(*pops[idx].chromosome[12:]))


if __name__ == "__main__":
    np.random.seed(2025)
    random.seed(2025)

    model = LLMEnergy(scene="Headwinds", rep="NZ")
    gam = GeneticAlgorithMulti(model, n_pop=200, epoch=1000, p_muta=0.2)
    gam()
