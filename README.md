# Laptop Prize Prediction

A machine learning project that predicts laptop prices using various regression models and provides an interactive Streamlit web application for model comparison and price prediction.

## Project Overview

This project uses multiple machine learning algorithms to predict laptop prices based on their features. It includes:
- Data preprocessing and exploration
- Training and evaluation of multiple models
- Model comparison and metrics
- Interactive web application for predictions

## Features

- **Multiple ML Models**: Linear Regression, KNN, Decision Tree, SVR, Random Forest, and AdaBoost
- **Streamlit Web App**: Interactive interface for:
  - Exploratory Data Analysis (EDA)
  - Model performance comparison
  - Real-time price predictions
- **Comprehensive Metrics**: R² score and Mean Absolute Error for model evaluation
- **Data Processing**: Automated data cleaning and feature engineering

## Project Structure

```
Laptop_prize_Prediction/
├── app.py                    # Streamlit web application
├── laptop.py                 # Model training and data preprocessing script
├── laptop_data.csv           # Dataset with laptop information
├── requirements.txt          # Python dependencies
├── models/                   # Trained model pipelines and metrics
│   ├── metrics.json         # Model performance metrics
│   └── *.pkl                # Trained model files
├── df.pkl                    # Processed dataframe
└── pipe.pkl                  # Data preprocessing pipeline
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rajan3081/Laptop-Prize-Prediction.git
cd Laptop-Prize-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Web App
```bash
streamlit run app.py
```

The application will open in your browser with three main sections:
- **Home/EDA**: Explore the dataset with visualizations
- **Model Comparison**: Compare performance metrics of different models
- **Prediction**: Make price predictions for new laptops

### Train Models
To retrain models with your data:
```bash
python laptop.py
```

## Dataset

The project uses `laptop_data.csv` containing laptop specifications and prices. Features include:
- Company
- TypeName
- RAM
- Weight
- TouchScreen
- IPS
- Screen Size
- Resolution (ppi)
- CPU
- GPU
- Price (target variable)

## Models Used

1. **Linear Regression**: Baseline model for price prediction
2. **K-Nearest Neighbors (KNN)**: n_neighbors=3
3. **Decision Tree**: max_depth=8
4. **Support Vector Regression (SVR)**: RBF kernel with C=10000
5. **Random Forest**: 100 estimators with optimized hyperparameters
6. **AdaBoost**: Gradient boosting regression

## Performance Metrics

Models are evaluated using:
- **R² Score**: Coefficient of determination
- **Mean Absolute Error (MAE)**: Average prediction error

Metrics are saved in `models/metrics.json` for easy comparison.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

See `requirements.txt` for specific versions.

## Author

Rajan3081

## License

This project is open source and available for educational purposes.
