# Research-Project

This repository contains all code and resources for my Data Science MSc research project. The project focuses on analyzing building-related data, with a particular emphasis on estimating Energy Use Intensity (EUI) through data cleaning, transformation, and exploratory data analysis.


## Project Overview

The goal of this is to prepare and analyse data relevant to energy usage in commercial buildings. This involves:

- Data cleaning and preprocessing
- Feature extraction and EUI metric generation
- Exploratory data analysis
- Experimental design and trials

The final aim is to support Bayesian Machine Learning modeling for EUI estimation, as part of my research.


## Repository Structure

<pre lang="markdown">
```
Research-Project/
│
├── Data/                    # Raw and cleaned datasets
├── Add_EUI.ipynb           # Adds EUI-related metrics to the dataset
├── Data_Cleaning.ipynb     # Data cleaning and preprocessing steps
├── trial.ipynb             # Experimental trial notebook
└── README.md               # This file
```
</pre>


## Getting Started

To run this project locally, follow the instructions below.

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Jupyter Notebook or JupyterLab
- pip or conda package manager

### Installation

1. **Clone the repository:**

```
bash
git clone https://github.com/georgepaul171/Research-Project.git
cd Research-Project
```

2. **Optional Create a Virtual Environment:**

```
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows
```

3. **Install the Required Packages:**

Use the requirements.txt file


## Usage

1. **Data Cleaning:**

   Open `Data_Cleaning.ipynb` to explore how missing values, duplicates, and formatting inconsistencies are handled.

2. **Add EUI:**

   Use `Add_EUI.ipynb` to calculate and merge Energy Use Intensity metrics based on available features.

3. **Run Experiments:**

   Use `trial.ipynb` to try out initial visualizations, feature correlations, and modeling ideas.

Ensure that your datasets are placed in the `Data/` folder and that paths are correctly referenced inside each notebook.


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
