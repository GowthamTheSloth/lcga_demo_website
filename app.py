# app.py
from flask import Flask, render_template, request, jsonify
from algorithms.simple_ga import SimpleGA
from algorithms.lcga import LCGA
from algorithms.de import DifferentialEvolution
from algorithms.pso import PSO
from utils.benchmarks import FUNCTIONS
import time
import traceback
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Default demo settings (fast)
DEFAULTS = {
    "dim": 10,
    "pop_size": 60,
    "generations": 80
}

# Parameter limits for safety
MAX_DIM = 50
MAX_POP_SIZE = 500
MAX_GENERATIONS = 500

@app.route("/")
def index():
    functions = sorted(list(FUNCTIONS.keys()))
    return render_template("index.html", functions=functions)

@app.route("/run", methods=["POST"])
def run():
    try:
        data = request.json or {}
        func_name = data.get("function", "Rastrigin")
        dim = int(data.get("dim", DEFAULTS["dim"]))
        pop_size = int(data.get("pop_size", DEFAULTS["pop_size"]))
        generations = int(data.get("generations", DEFAULTS["generations"]))

        # Validate parameters
        if func_name not in FUNCTIONS:
            return jsonify({"error": f"Unknown function: {func_name}. Available: {list(FUNCTIONS.keys())}"}), 400
        
        if dim < 2 or dim > MAX_DIM:
            return jsonify({"error": f"Dimension must be between 2 and {MAX_DIM}"}), 400
        
        if pop_size < 10 or pop_size > MAX_POP_SIZE:
            return jsonify({"error": f"Population size must be between 10 and {MAX_POP_SIZE}"}), 400
        
        if generations < 5 or generations > MAX_GENERATIONS:
            return jsonify({"error": f"Generations must be between 5 and {MAX_GENERATIONS}"}), 400

        func = FUNCTIONS[func_name]

        # Collect results for all algorithms
        algos = []

        # GA
        ga = SimpleGA(func, dim=dim, pop_size=pop_size, generations=generations)
        t0 = time.time()
        ga_res = ga.run()
        t_ga = time.time() - t0
        algos.append({
            "name": "GA",
            "best_fitness": ga_res["best_fitness"],
            "time_sec": round(t_ga, 3),
            "history": ga_res["history"],
            "diversity_history": ga_res.get("diversity_history", []),
            "age_history": ga_res.get("age_history", []),
            "genotype_diversity": ga_res.get("genotype_diversity", []),
        })

        # LCGA
        lcga = LCGA(func, dim=dim, pop_size=pop_size, generations=generations)
        t0 = time.time()
        lcga_res = lcga.run()
        t_lcga = time.time() - t0
        algos.append({
            "name": "LCGA",
            "best_fitness": lcga_res["best_fitness"],
            "time_sec": round(t_lcga, 3),
            "history": lcga_res["history"],
            "diversity_history": lcga_res.get("diversity_history", []),
            "age_history": lcga_res.get("age_history", []),
            "genotype_diversity": lcga_res.get("genotype_diversity", []),
        })

        # Differential Evolution (DE)
        de = DifferentialEvolution(func, dim=dim, pop_size=pop_size, generations=generations)
        t0 = time.time()
        de_res = de.run()
        t_de = time.time() - t0
        algos.append({
            "name": "DE",
            "best_fitness": de_res["best_fitness"],
            "time_sec": round(t_de, 3),
            "history": de_res["history"],
            "diversity_history": de_res.get("diversity_history", []),
            "age_history": de_res.get("age_history", []),
            "genotype_diversity": de_res.get("genotype_diversity", []),
        })

        # Particle Swarm Optimization (PSO)
        pso = PSO(func, dim=dim, pop_size=pop_size, generations=generations)
        t0 = time.time()
        pso_res = pso.run()
        t_pso = time.time() - t0
        algos.append({
            "name": "PSO",
            "best_fitness": pso_res["best_fitness"],
            "time_sec": round(t_pso, 3),
            "history": pso_res["history"],
            "diversity_history": pso_res.get("diversity_history", []),
            "age_history": pso_res.get("age_history", []),
            "genotype_diversity": pso_res.get("genotype_diversity", []),
        })

        # Back-compat improvement percent (LCGA vs GA)
        improvement = None
        try:
            improvement = 100.0 * (lcga_res["best_fitness"] - ga_res["best_fitness"]) / max(1e-12, abs(ga_res["best_fitness"]))
        except Exception:
            improvement = None

        res = {
            "function": func_name,
            "algorithms": algos,
            "improvement_percent": None if improvement is None else round(improvement, 3),
            "parameters": {
                "dimension": dim,
                "population_size": pop_size,
                "generations": generations
            }
        }
        return jsonify(res)
    
    except Exception as e:
        app.logger.error(f"Error in run endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
