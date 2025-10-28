# algorithms/lcga.py
import numpy as np
import random

class LCGA:
    """
    Compact Life-Cycle Genetic Algorithm.
    Returns best_history list (best fitness each generation).
    This is a simplified LCGA tuned for interactive demos.
    """

    class Individual:
        def __init__(self, chrom, fitness=None, age=0):
            self.chrom = np.array(chrom, dtype=float)
            self.fitness = fitness
            self.age = age

    def __init__(self, func, dim=10, pop_size=60, bounds=(-5.12,5.12),
                 max_age=8, maturity_age=2, birth_rate=0.3,
                 crossover_rate=0.85, mutation_rate=0.12, generations=100, seed=1):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.max_age = max_age
        self.maturity_age = maturity_age
        self.birth_rate = birth_rate
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.generations = generations
        random.seed(seed)
        np.random.seed(seed)
        self.population = self._init_population()
        
        # Track diversity and age history
        self.diversity_history = []
        self.age_history = []
        self.genotype_diversity = []

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            ind = LCGA.Individual(chrom, fitness=self.func(chrom), age=random.randint(0, self.maturity_age))
            pop.append(ind)
        return pop

    def _evaluate(self, ind):
        ind.fitness = float(self.func(ind.chrom))

    def _death_prob(self, age):
        # quadratic increasing death prob (paper-like)
        base = (age / max(1.0, self.max_age))
        # Soften early deaths to preserve diversity
        return float(base ** 1.2) * 0.8

    def _select_parent(self, k=3):
        sample = random.sample(self.population, min(k, len(self.population)))
        parent = min(sample, key=lambda ind: ind.fitness)
        return parent

    def _crossover_blend(self, p1, p2):
        alpha = 0.5
        c1 = alpha*p1.chrom + (1-alpha)*p2.chrom
        c2 = alpha*p2.chrom + (1-alpha)*p1.chrom
        return LCGA.Individual(c1), LCGA.Individual(c2)

    def _mutate(self, ind):
        # Age-adaptive mutation: boost for young individuals
        mr = self.mr
        if getattr(ind, 'age', 0) < self.maturity_age:
            mr = min(1.0, mr * 1.8)
        for i in range(self.dim):
            if random.random() < mr:
                ind.chrom[i] += np.random.normal(0, 0.3)
                ind.chrom[i] = np.clip(ind.chrom[i], self.bounds[0], self.bounds[1])
        return ind
    
    def _calculate_diversity(self):
        """Calculate population diversity as standard deviation of fitness values"""
        if len(self.population) < 2:
            return 0.0
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        if len(fitness_values) < 2:
            return 0.0
        return float(np.std(fitness_values))
    
    def _calculate_mean_age(self):
        """Calculate mean age of population"""
        if not self.population:
            return 0.0
        ages = [ind.age for ind in self.population]
        return float(np.mean(ages))

    def run(self):
        best_history = []
        for step in range(self.generations):
            # Age increment
            for ind in self.population:
                ind.age += 1

            # Death phase (asynchronous style)
            survivors = []
            for ind in self.population:
                if ind.age > self.max_age:
                    continue
                if random.random() < self._death_prob(ind.age):
                    continue
                survivors.append(ind)
            self.population = survivors

            # Reproduction phase: pick mature parents and create offspring
            mature = [ind for ind in self.population if ind.age >= self.maturity_age]
            offspring = []
            births = max(1, int(self.birth_rate * self.pop_size))
            for _ in range(births):
                if len(mature) < 2:
                    break
                p1, p2 = random.sample(mature, 2)
                if random.random() < self.cr:
                    c1, c2 = self._crossover_blend(p1, p2)
                else:
                    c1, c2 = LCGA.Individual(p1.chrom.copy()), LCGA.Individual(p2.chrom.copy())
                c1.age = 0; c2.age = 0
                offspring.extend([self._mutate(c1), self._mutate(c2)])

            # Evaluate offspring and add
            for child in offspring:
                self._evaluate(child)
            self.population.extend(offspring)

            # If population too large, keep best
            if len(self.population) > self.pop_size:
                self.population = sorted(self.population, key=lambda ind: ind.fitness)[:self.pop_size]

            # If extinct, reinitialize
            if len(self.population) == 0:
                self.population = self._init_population()

            # Evaluate (ensure all have fitness)
            for ind in self.population:
                if ind.fitness is None:
                    self._evaluate(ind)

            # Calculate and store diversity and mean age
            diversity = self._calculate_diversity()
            mean_age = self._calculate_mean_age()
            self.diversity_history.append(diversity)
            self.age_history.append(mean_age)
            # Calculate genotype diversity: mean per-gene std
            if len(self.population) > 0:
                chroms = np.array([ind.chrom for ind in self.population])
                if chroms.shape[0] > 1:
                    per_gene_std = np.std(chroms, axis=0)
                    geno_div = float(np.mean(per_gene_std))
                else:
                    geno_div = 0.0
            else:
                geno_div = 0.0
            self.genotype_diversity.append(geno_div)
            
            # Print every 10 generations
            if (step + 1) % 10 == 0:
                print(f"Gen {step + 1}: mean_age={mean_age:.2f}, diversity={diversity:.4f}")

            # Record best
            best = min(self.population, key=lambda ind: ind.fitness)
            best_history.append(float(best.fitness))

        best = min(self.population, key=lambda ind: ind.fitness)
        return {"best_fitness": float(best.fitness),
                "best_chrom": best.chrom.tolist(),
                "history": best_history,
                "diversity_history": self.diversity_history,
                "age_history": self.age_history,
                "genotype_diversity": self.genotype_diversity}
