# French Motor Insurance Coding Challenge

This repository contains a series of Jupyter notebooks for the analysis and modeling of insurance claims data, specifically focused on French motor insurance. The project explores various data analysis techniques and machine learning models to predict insurance claims over exposure time.

## Project Structure

- `exploratory_data_analysis.ipynb`: This notebook contains the exploratory data analysis (EDA) of the dataset. It includes:
  - Importing and cleaning the data.
  - Analyzing the distribution of key features.
  - Handling missing values and duplicates.
  - Examining categorical and numerical variables.
  - Identifying potential issues such as multicollinearity and non-linear relationships.

- `modeling_notebook.ipynb`: This notebook focuses on the manual modeling process. It includes:
  - Implementing various regression models, starting with linear regression and moving to more complex non-linear models.
  - Evaluating model performance and discussing the challenges faced due to low correlations identified during the EDA.
  - Suggestions for further improvements, including feature engineering and the use of advanced modeling techniques.

## Getting Started

To get started with the notebooks:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/insurance-coding-challenge.git
   cd insurance-coding-challenge
   ```

2. **Set up a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Open the notebooks**:
   You can open the notebooks using Jupyter by running:
   ```bash
   jupyter notebook
   ```
   Then, navigate to `exploratory_data_analysis.ipynb` and `modeling_notebook.ipynb` to start exploring the data and models.

## Requirements

The `requirements.txt` file contains all the necessary dependencies to run the notebooks. Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Overview

This project aims to:

- Explore the relationship between various features in the dataset and their impact on insurance claims.
- Build predictive models to estimate insurance claim amounts over exposure time, taking into account potential challenges such as low correlation between features and the target variable.
- Experiment with different feature engineering techniques and model optimization strategies to improve model performance.

## Future Work

There are several additional steps that could be taken to improve the models and analysis:

- **Feature Engineering**: Collect more features such as policyholder demographics, vehicle characteristics, and geographic data.
- **Reframe the Problem**: Consider reframing the problem as a classification task rather than regression.
- **Incorporate Domain Knowledge**: Consult with insurance experts to incorporate domain knowledge into the model design.
- **Advanced Modeling**: Explore advanced modeling techniques such as ensembling and stacking models.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
