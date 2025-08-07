# Box Office Revenue Prediction Using Linear Regression

## Overview
This project aims to predict the domestic box office revenue of movies using machine learning models, primarily focusing on Linear Regression and comparing it with advanced models like XGBoost. The goal is to identify the key factors influencing a movie's financial performance and build a model that estimates revenue before release.

This project is part of the "50 Days, 50 Projects – Machine Learning Challenge".

## Objectives
- Build a predictive model to estimate domestic box office revenue.
- Understand the most influential features affecting revenue.
- Apply data preprocessing, feature engineering, visualization, and model evaluation techniques.
- Compare linear regression with XGBoost for predictive performance.

## Dataset
The dataset contains information about:
- Movie titles
- Genres
- MPAA ratings
- Number of opening theaters
- Release days
- Distributor
- Revenue data (domestic, world, opening)

Data source: `boxoffice.csv`

## Libraries Used
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- XGBoost

## Steps Followed

### 1. Data Loading and Exploration
- Loaded the dataset using pandas.
- Checked data types, missing values, and overall structure.

### 2. Data Cleaning and Preprocessing
- Dropped highly missing or irrelevant columns (`world_revenue`, `opening_revenue`, `budget`).
- Filled missing values for `MPAA` and `genres` with the mode.
- Converted columns with currency and commas to numeric.
- Label encoded categorical variables (`MPAA`, `distributor`).
- Performed one-hot encoding on genres using `CountVectorizer`, and removed rare genres.

### 3. Feature Engineering
- Applied log transformation to skewed numeric features.
- Removed highly correlated features.
- Normalized features using `StandardScaler`.

### 4. Model Building
- Used `train_test_split` for 90/10 training and validation sets.
- Trained the `XGBRegressor` model.
- Evaluated performance using Mean Absolute Error (MAE).

### 5. Results
- **Training MAE**: 0.21
- **Validation MAE**: 0.63

The model performed well on training data, but the validation MAE indicates potential overfitting.

## Key Insights
- PG and R rated movies generally earned more revenue.
- Log transformation and feature scaling significantly improved model stability.
- XGBoost outperformed linear regression in predictive accuracy.


## Future Improvements
- Incorporate additional features like actor popularity, director track record, and social media sentiment.
- Test on a larger dataset for better generalization.
- Apply regularization to reduce overfitting.

## Author
Aparna Ashok – as part of the 50 Days, 50 Projects ML Challenge.

