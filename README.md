# Research-Project

This repository contains all code and resources for my Data Science MSc research project. The project focuses on analyzing building-related data, with a particular emphasis on estimating Energy Use Intensity (EUI) through data cleaning, transformation, exploratory data analysis, and Bayesian modeling approaches.

## Project Overview

The objective is to prepare analyse, and model data relevant to energy usage in commercial office buildings. Key tasks include:

- Data cleaning and preprocessing
- Feature extraction and EUI metric generation
- Exploratory data analysis (EDA)
- Bayesian hierarchical modeling
- Bayesian sparse regression modeling

This project ultimately supports Bayesian Machine Learning techniques to estimate EUI, as part of my MSc dissertation.


## Repository Structure

<pre lang="markdown">
```
Research-Project/
├── Data/                  # Raw & cleaned datasets
├── analysis-cleaning/     # Data cleaning & EDA notebooks + scripts
├── hierarchical/          # Bayesian hierarchical models
└── sparse‑regression/     # Bayesian sparse regression models
```
</pre>


## Getting Started

### Prerequisites

Ensure the following are installed:

- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip or conda

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/georgepaul171/Research-Project.git
cd Research-Project


2. **Optional Create a Virtual Environment:**

```
python -m venv env
source env/bin/activate      # macOS/Linux
# or
env\Scripts\activate         # Windows
```

3. **Install the Required Packages:**

```
pip install -r requirements.txt
```

## Usage

### 1. **Data Cleaning & EDA**
Open notebooks in the `analysis-cleaning/` directory to explore data handling:

- `add_EUI_SQMT.ipynb`: Calculates EUI per square meter  
- `Filtered_Data_Analysis.ipynb`: EDA on clean datasets  
- Graphs are stored under `Graphs/`

### 2. **Modeling**

- Run `bay_hier.ipynb` for hierarchical Bayesian modeling  
- Use `bay_sparse_reg.ipynb` for sparse regression model experiments

### 3. **Datasets**
Place all relevant CSV and Excel files in the `Data/` folder.  
Ensure paths in notebooks point accordingly.


## Contributing

As this project is part of my individual MSc dissertation, I am required to complete the work solely on my own. 

For this reason, I am **not accepting external contributions** at this time.

If you have feedback or spot any issues, feel free to open an issue — but please understand that I cannot incorporate external code or direct contributions.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

- GitHub: [@georgepaul171](https://github.com/georgepaul171)
- Email: gp813@bath.ac.uk

---

**Note:** This project is a work-in-progress as part of an MSc dissertation. Expect frequent updates
