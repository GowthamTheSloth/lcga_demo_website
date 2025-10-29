import numpy as np
import random


class LCGA:
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
        self.diversity_history = []
        self.age_history = []
        self.genotype_diversity = []

    def _init_pop(self):
        self.population = [np.random.uniform(self.bounds[0], self.bounds[1], self.dim) for _ in range(self.pop_size)]
        self.fitness = [self.func(ind) for ind in self.population]
        self.ages = [0 for _ in range(self.pop_size)]
        self.lifespans = [random.randint(8, 20) for _ in range(self.pop_size)]

    def _stage(self, age, lifespan):
        if lifespan <= 0:
            return "senior"
        r = age / float(lifespan)
        if r < 0.3:
            return "juvenile"
        if r < 0.8:
            return "adult"
        return "senior"

    def _tournament_lifecycle(self, k=3):
        ids = random.sample(range(self.pop_size), k)
        def score(i):
            s = self._stage(self.ages[i], self.lifespans[i])
            if s == "juvenile":
                w = 0.9
            elif s == "adult":
                w = 1.0
            else:
                w = 1.1
            return self.fitness[i] * w
        best = min(ids, key=score)
        return best

    def _blend_crossover(self, a, b, alpha=0.5):
        c1 = alpha * a + (1 - alpha) * b
        c2 = alpha * b + (1 - alpha) * a
        return c1, c2

    def _mutate(self, x, sigma=0.2):
        for i in range(self.dim):
            if random.random() < self.mr:
                x[i] += np.random.normal(0, sigma)
                x[i] = np.clip(x[i], self.bounds[0], self.bounds[1])
        return x

    def _stage_adjusted_rates(self, i, j):
        si = self._stage(self.ages[i], self.lifespans[i])
        sj = self._stage(self.ages[j], self.lifespans[j])
        cr = self.cr
        mr = self.mr
        if si == "juvenile" or sj == "juvenile":
            cr *= 0.8
            mr *= 1.5
        if si == "adult" or sj == "adult":
            cr *= 1.1
        if si == "senior" or sj == "senior":
            mr *= 0.8
        return min(1.0, cr), min(1.0, mr)

    def _calculate_diversity(self):
        if len(self.fitness) < 2:
            return 0.0
        return float(np.std(self.fitness))

    def _apply_mortality(self, new_pop, new_ages, new_lifespans):
        survivors = []
        s_ages = []
        s_lifespans = []
        for x, a, L in zip(new_pop, new_ages, new_lifespans):
            dead = a >= L
            if not dead and self._stage(a, L) == "senior" and random.random() < 0.25:
                dead = True
            if dead:
                nx = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                survivors.append(nx)
                s_ages.append(0)
                s_lifespans.append(random.randint(8, 20))
            else:
                survivors.append(x)
                s_ages.append(a)
                s_lifespans.append(L)
        return survivors, s_ages, s_lifespans

    def run(self):
        history = []
        for g in range(self.gen):
            best_idx = int(np.argmin(self.fitness))
            elite = self.population[best_idx].copy()
            elite_age = self.ages[best_idx] + 1
            elite_L = self.lifespans[best_idx]

            new_pop = [elite]
            new_ages = [elite_age]
            new_Ls = [elite_L]

            while len(new_pop) < self.pop_size:
                i = self._tournament_lifecycle()
                j = self._tournament_lifecycle()
                while j == i:
                    j = self._tournament_lifecycle()

                cr, base_mr = self._stage_adjusted_rates(i, j)
                ai = self.population[i].copy()
                bj = self.population[j].copy()

                if random.random() < cr:
                    c1, c2 = self._blend_crossover(ai, bj, alpha=0.5)
                else:
                    c1, c2 = ai, bj

                old_mr = self.mr
                self.mr = base_mr
                c1 = self._mutate(c1, sigma=0.25)
                c2 = self._mutate(c2, sigma=0.25)
                self.mr = old_mr

                new_pop.append(c1)
                new_ages.append(0)
                new_Ls.append(random.randint(8, 20))
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
                    new_ages.append(0)
                    new_Ls.append(random.randint(8, 20))

            new_pop, new_ages, new_Ls = self._apply_mortality(new_pop, new_ages, new_Ls)

            self.population = new_pop
            self.ages = [a + 1 for a in new_ages]
            self.lifespans = new_Ls
            self.fitness = [self.func(ind) for ind in self.population]

            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            mean_age = float(np.mean(self.ages)) if len(self.ages) > 0 else 0.0
            self.age_history.append(mean_age)
            if len(self.population) > 1:
                chroms = np.array(self.population)
                per_gene_std = np.std(chroms, axis=0)
                geno_div = float(np.mean(per_gene_std))
            else:
                geno_div = 0.0
            self.genotype_diversity.append(geno_div)

            best = min(self.fitness)
            history.append(float(best))
            if (g + 1) % 10 == 0:
                print(f"LCGA Gen {g + 1}: diversity={diversity:.4f}")

        best_idx = int(np.argmin(self.fitness))
        return {
            "best_fitness": float(self.fitness[best_idx]),
            "best_chrom": self.population[best_idx].tolist(),
            "history": history,
            "diversity_history": self.diversity_history,
            "age_history": self.age_history,
            "genotype_diversity": self.genotype_diversity,
        }

