# algorithms/simple_ga.py
import numpy as np
import random

class SimpleGA:
    """
    Minimal generational GA for continuous optimization.
    Returns best_history list (best fitness each generation).
    """

    def __init__(self, func, dim=10, pop_size=60, bounds=(-5.12, 5.12),
                 crossover_rate=0.9, mutation_rate=0.1, generations=100, seed=1):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.gen = generations
        random.seed(seed)
        np.random.seed(seed)
        self._init_pop()
        
        # Track diversity history
        self.diversity_history = []
        # Track mean age history
        self.age_history = []
        # Ages aligned with population
        self.ages = [0 for _ in range(self.pop_size)]
        # Genotype diversity (mean per-gene std) per generation
        self.genotype_diversity = []

    def _init_pop(self):
        self.population = [np.random.uniform(self.bounds[0], self.bounds[1], self.dim) for _ in range(self.pop_size)]
        self.fitness = [self.func(ind) for ind in self.population]

    def _tournament(self, k=3):
        ids = random.sample(range(self.pop_size), k)
        best = min(ids, key=lambda i: self.fitness[i])
        return self.population[best].copy()

    def _crossover(self, a, b):
        # blend crossover (simple)
        alpha = 0.5
        c1 = alpha*a + (1-alpha)*b
        c2 = alpha*b + (1-alpha)*a
        return c1, c2

    def _mutate(self, x):
        for i in range(self.dim):
            if random.random() < self.mr:
                x[i] += np.random.normal(0, 0.2)
                x[i] = np.clip(x[i], self.bounds[0], self.bounds[1])
        return x
    
    def _calculate_diversity(self):
        """Calculate population diversity as standard deviation of fitness values"""
        if len(self.fitness) < 2:
            return 0.0
        return float(np.std(self.fitness))

    def run(self):
        best_history = []
        for g in range(self.gen):
            # Elitism: carry over best individual with age increment
            best_idx_current = int(np.argmin(self.fitness))
            elite = self.population[best_idx_current].copy()
            elite_age = self.ages[best_idx_current] + 1

            new_pop = [elite]
            new_ages = [elite_age]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                if random.random() < self.cr:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(self._mutate(c1))
                new_ages.append(0)
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
                    new_ages.append(0)
            self.population = new_pop
            self.ages = new_ages
            self.fitness = [self.func(ind) for ind in self.population]
            
            # Calculate and store diversity
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            # Calculate and store mean age
            mean_age = float(np.mean(self.ages)) if len(self.ages) > 0 else 0.0
            self.age_history.append(mean_age)
            # Calculate genotype diversity: mean per-gene std
            if len(self.population) > 1:
                chroms = np.array(self.population)
                per_gene_std = np.std(chroms, axis=0)
                geno_div = float(np.mean(per_gene_std))
            else:
                geno_div = 0.0
            self.genotype_diversity.append(geno_div)
            
            # Print every 10 generations
            if (g + 1) % 10 == 0:
                print(f"GA Gen {g + 1}: diversity={diversity:.4f}")
            
            best = min(self.fitness)
            best_history.append(float(best))
        best_idx = int(np.argmin(self.fitness))
        return {"best_fitness": float(self.fitness[best_idx]),
                "best_chrom": self.population[best_idx].tolist(),
                "history": best_history,
                "diversity_history": self.diversity_history,
                "age_history": self.age_history,
                "genotype_diversity": self.genotype_diversity}
