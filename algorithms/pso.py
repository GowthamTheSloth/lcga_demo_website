import numpy as np
import random

class PSO:
    def __init__(self, func, dim=10, pop_size=60, bounds=(-5.12, 5.12), generations=100,
                 w=0.7, c1=1.5, c2=1.5, vmax=None, seed=1):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.gen = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax
        random.seed(seed)
        np.random.seed(seed)
        self._init_swarm()
        self.diversity_history = []
        self.genotype_diversity = []
        self.age_history = []

    def _init_swarm(self):
        low, high = self.bounds
        self.positions = [np.random.uniform(low, high, self.dim) for _ in range(self.pop_size)]
        vel_scale = (high - low) * 0.1
        self.velocities = [np.random.uniform(-vel_scale, vel_scale, self.dim) for _ in range(self.pop_size)]
        self.pbest_pos = [p.copy() for p in self.positions]
        self.pbest_fit = [self.func(p) for p in self.positions]
        g_idx = int(np.argmin(self.pbest_fit))
        self.gbest_pos = self.pbest_pos[g_idx].copy()
        self.gbest_fit = float(self.pbest_fit[g_idx])

    def _clip(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def _calculate_diversity(self, fits):
        if len(fits) < 2:
            return 0.0
        return float(np.std(fits))

    def run(self):
        best_history = []
        for _ in range(self.gen):
            new_fits = []
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_pos - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                if self.vmax is not None:
                    self.velocities[i] = np.clip(self.velocities[i], -self.vmax, self.vmax)
                self.positions[i] = self._clip(self.positions[i] + self.velocities[i])
                f = self.func(self.positions[i])
                new_fits.append(f)
                if f < self.pbest_fit[i]:
                    self.pbest_fit[i] = f
                    self.pbest_pos[i] = self.positions[i].copy()
            g_idx = int(np.argmin(self.pbest_fit))
            if self.pbest_fit[g_idx] < self.gbest_fit:
                self.gbest_fit = float(self.pbest_fit[g_idx])
                self.gbest_pos = self.pbest_pos[g_idx].copy()

            # metrics
            diversity = self._calculate_diversity(new_fits)
            self.diversity_history.append(diversity)
            if len(self.positions) > 1:
                chroms = np.array(self.positions)
                per_gene_std = np.std(chroms, axis=0)
                geno_div = float(np.mean(per_gene_std))
            else:
                geno_div = 0.0
            self.genotype_diversity.append(geno_div)
            self.age_history.append(0.0)

            best_history.append(float(self.gbest_fit))

        return {
            "best_fitness": float(self.gbest_fit),
            "best_chrom": self.gbest_pos.tolist(),
            "history": best_history,
            "diversity_history": self.diversity_history,
            "age_history": self.age_history,
            "genotype_diversity": self.genotype_diversity,
        }
