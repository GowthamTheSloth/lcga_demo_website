# Life-Cycle Genetic Algorithm (LCGA) Demo

A web-based interactive demonstration comparing traditional Genetic Algorithm (GA) with Life-Cycle Genetic Algorithm (LCGA) on benchmark optimization functions.

## Features

- **Interactive Web Interface**: Modern, responsive design with real-time parameter adjustment
- **Algorithm Comparison**: Side-by-side comparison of GA vs LCGA performance
- **Multiple Benchmark Functions**: Sphere, Rosenbrock, Rastrigin, Ackley, and Griewank functions
- **Real-time Visualization**: Dynamic plots showing convergence over generations
- **Performance Metrics**: Execution time and improvement percentage calculations

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. **Clone or download this project** to your local machine

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install Flask==2.3.3 numpy==1.24.3
   ```

## Running the Application

1. **Navigate to the project directory**:
   ```bash
   cd "C:\Users\hp\OneDrive\Desktop\SCEAI_projec t"
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```

3. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Select a benchmark function** from the dropdown menu
2. **Adjust parameters**:
   - **Dimension**: Problem dimensionality (2-50)
   - **Population Size**: Number of individuals (10-500)
   - **Generations**: Number of iterations (5-500)
3. **Click "Run Comparison"** to execute both algorithms
4. **View results**:
   - Summary table with performance metrics
   - Convergence plot showing algorithm progress
   - Raw data output

## Algorithm Parameters

### Default Settings (Fast Demo)
- Dimension: 10
- Population Size: 60
- Generations: 80

### Parameter Limits
- Dimension: 2-50 (higher values may be slow)
- Population Size: 10-500
- Generations: 5-500

## Benchmark Functions

1. **Sphere**: Simple unimodal function
2. **Rosenbrock**: Classic optimization problem
3. **Rastrigin**: Multimodal function with many local minima
4. **Ackley**: Complex multimodal function
5. **Griewank**: Product term creates complexity

## Troubleshooting

### Common Issues

1. **Port already in use**:
   - Change the port in `app.py`: `app.run(debug=True, port=5001)`

2. **Module not found errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

3. **Slow performance**:
   - Reduce dimension, population size, or generations
   - Close other applications to free up system resources

4. **Browser compatibility**:
   - Use modern browsers (Chrome, Firefox, Safari, Edge)
   - Enable JavaScript

### Performance Tips

- Start with smaller parameters for testing
- Increase parameters gradually for better results
- Monitor system resources during execution

## Technical Details

### Architecture
- **Backend**: Flask web framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Plotly.js
- **Algorithms**: Custom Python implementations

### File Structure
```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── algorithms/
│   ├── simple_ga.py      # Traditional GA implementation
│   └── lcga.py          # Life-Cycle GA implementation
├── utils/
│   └── benchmarks.py    # Benchmark function definitions
├── templates/
│   └── index.html       # Web interface template
└── static/
    └── style.css        # Styling and responsive design
```

## Development

To modify or extend the application:

1. **Add new benchmark functions** in `utils/benchmarks.py`
2. **Modify algorithm parameters** in the respective algorithm files
3. **Update the web interface** in `templates/index.html`
4. **Customize styling** in `static/style.css`

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please check the troubleshooting section above.

