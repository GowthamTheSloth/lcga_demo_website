import numpy as np
import random

class DifferentialEvolution:
    def __init__(self, func, dim=10, pop_size=60, bounds=(-5.12, 5.12), generations=100, F=0.5, Cr=0.9, seed=1):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.gen = generations
        self.F = F
        self.Cr = Cr
        random.seed(seed)
        np.random.seed(seed)
        self._init_pop()
        self.diversity_history = []
        self.genotype_diversity = []
        self.age_history = []

    def _init_pop(self):
        self.population = [np.random.uniform(self.bounds[0], self.bounds[1], self.dim) for _ in range(self.pop_size)]
        self.fitness = [self.func(ind) for ind in self.population]

    def _mutant_vector(self, idx):
        idxs = list(range(self.pop_size))
        idxs.remove(idx)
        a, b, c = random.sample(idxs, 3)
        va = self.population[a]
        vb = self.population[b]
        vc = self.population[c]
        v = va + self.F * (vb - vc)
        return np.clip(v, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant):
        j_rand = random.randrange(self.dim)
        trial = target.copy()
        for j in range(self.dim):
            if random.random() < self.Cr or j == j_rand:
                trial[j] = mutant[j]
        return np.clip(trial, self.bounds[0], self.bounds[1])

    def _calculate_diversity(self):
        if len(self.fitness) < 2:
            return 0.0
        return float(np.std(self.fitness))

    def run(self):
        best_history = []
        for g in range(self.gen):
            new_pop = []
            new_fit = []
            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self._mutant_vector(i)
                trial = self._crossover(target, mutant)
                f_trial = self.func(trial)
                if f_trial <= self.fitness[i]:
                    new_pop.append(trial)
                    new_fit.append(f_trial)
                else:
                    new_pop.append(target)
                    new_fit.append(self.fitness[i])
            self.population = new_pop
            self.fitness = new_fit

            # metrics
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            if len(self.population) > 1:
                chroms = np.array(self.population)
                per_gene_std = np.std(chroms, axis=0)
                geno_div = float(np.mean(per_gene_std))
            else:
                geno_div = 0.0
            self.genotype_diversity.append(geno_div)
            self.age_history.append(0.0)

            best = min(self.fitness)
            best_history.append(float(best))

        best_idx = int(np.argmin(self.fitness))
        return {
            "best_fitness": float(self.fitness[best_idx]),
            "best_chrom": self.population[best_idx].tolist(),
            "history": best_history,
            "diversity_history": self.diversity_history,
            "age_history": self.age_history,
            "genotype_diversity": self.genotype_diversity,
        }
