import mlflow 
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_ridge_regression_with_grid_search(X_train, X_test, y_train, y_test, data_preprocessing="binned", run_name="Ridge Regression with Grid Search"):
    with mlflow.start_run(run_name=run_name):
        try:
            # Log data preprocessing method
            mlflow.log_param("data_preprocessing", data_preprocessing)

            # Define the model and parameter grid
            model = Ridge(random_state=42)
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10, 100],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

            # Set up the Grid Search
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, 
                                       scoring='neg_mean_squared_error', n_jobs=1)

            # Fit the grid search
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Make predictions
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log best parameters
            mlflow.log_params(grid_search.best_params_)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Log the model
            mlflow.sklearn.log_model(best_model, "ridge_model")

            # Print results
            print("Ridge Regression with Grid Search Results:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"MSE: {mse}")
            print(f"R Squared: {r2}")

            # Log and print feature coefficients
            print("Model Coefficients:")
            for feature, coef in zip(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]), best_model.coef_):
                mlflow.log_metric(f"coef_{feature}", coef)
                print(f"{feature}: {coef}")

            # Log intercept
            mlflow.log_metric("intercept", best_model.intercept_)
            print(f"Intercept: {best_model.intercept_}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            mlflow.log_param("error", str(e))
            raise


def train_linear_regression(X_train, X_test, y_train, y_test, data_preprocessing="binned", run_name="Linear Regression"):
    with mlflow.start_run(run_name=run_name):
        # Log data preprocessing method
        mlflow.log_param("data_preprocessing", data_preprocessing)
        
        # Train linear regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Predict on test set
        y_pred = lr.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric('r2', r2)

        # Log model
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Print results
        print("Linear Regression Results:")
        print(f"MSE: {mse}")
        print(f"R Squared: {r2}")

        # Log feature coefficients
        for feature, coef in zip(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]), lr.coef_):
            mlflow.log_metric(f"coef_{feature}", coef)
            print(f"{feature}: {coef}")

        # Log intercept
        mlflow.log_metric("intercept", lr.intercept_)
        print(f"Intercept: {lr.intercept_}")



def train_poly_ridge_regression_optimized(X, y, data_preprocessing="binned", run_name="Optimized Poly Ridge Regression"):
    with mlflow.start_run(run_name=run_name):
        try:
            # Log data preprocessing method and model type
            mlflow.log_param("data_preprocessing", data_preprocessing)
            mlflow.log_param("model_type", "Optimized Polynomial Ridge Regression")

            # Split the data for initial tuning
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.3, random_state=42)

            # Define the pipeline
            poly_ridge_pipeline = Pipeline([
                ('poly', PolynomialFeatures(include_bias=False)),
                ('ridge', Ridge(random_state=42))
            ])

            # Define the parameter distributions
            param_distributions = {
                'poly__degree': randint(1, 4),  # Uniform integer 1 to 3
                'ridge__alpha': uniform(0.1, 100)  # Uniform float 0.1 to 100.1
            }

            # Create the randomized search object with early stopping
            random_search = RandomizedSearchCV(
                poly_ridge_pipeline,
                param_distributions,
                n_iter=20,  # Number of parameter settings sampled
                cv=3,  # Reduce number of folds
                scoring='neg_mean_squared_error',
                n_jobs=-1,  # Use all available cores
                random_state=42
            )

            # Fit the randomized search on the sample
            random_search.fit(X_sample, y_sample)

            # Get the best model
            best_model = random_search.best_estimator_

            # Retrain the best model on the full dataset
            best_model.fit(X, y)

            # Make predictions
            y_pred = best_model.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Log parameters
            mlflow.log_params(random_search.best_params_)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            # Log the best model
            mlflow.sklearn.log_model(best_model, "optimized_poly_ridge_model")

            # Print results
            print("Optimized Polynomial Ridge Regression Results:")
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Mean Squared Error: {mse}")
            print(f"R Squared: {r2}")

            # Examine the coefficients
            poly_features = best_model.named_steps['poly']
            ridge_model = best_model.named_steps['ridge']
            feature_names = poly_features.get_feature_names_out(X.columns if hasattr(X, 'columns') else None)
            coefficients = ridge_model.coef_

            # Log and print feature coefficients
            print("Model Coefficients:")
            for name, coef in zip(feature_names, coefficients):
                mlflow.log_metric(f"coef_{name}", coef)
                print(f"{name}: {coef}")

            # Log intercept
            mlflow.log_metric("intercept", ridge_model.intercept_)
            print(f"Intercept: {ridge_model.intercept_}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

    return best_model, mse, r2

def train_random_forest_optimized(X, y, data_type, run_name="Optimized Random Forest"):
    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_param("model_type", "Optimized Random Forest")
            mlflow.log_param("data_preprocessing", data_type)

            # Split the data for initial tuning
            X_sample, X_test, y_sample, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define the parameter distributions
            param_distributions = {
                "n_estimators": randint(50, 300),
                "max_depth": randint(5, 30),
                "min_samples_split": randint(2, 11),
                "max_features": uniform(0.1, 0.9)
            }

            rf = RandomForestRegressor(random_state=42)

            # Create the randomized search object
            random_search = RandomizedSearchCV(
                rf, 
                param_distributions, 
                n_iter=20,  # Number of parameter settings sampled
                cv=3,  # Reduce number of folds
                scoring='neg_mean_squared_error',
                n_jobs=-1,  # Use all available cores
                random_state=42
            )

            # Fit the randomized search on the sample
            random_search.fit(X_sample, y_sample)
            best_rf = random_search.best_estimator_

            # Retrain the best model on the full dataset
            best_rf.fit(X, y)

            # Make predictions
            y_pred = best_rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(best_rf, "optimized_random_forest_model")

            print("Optimized Random Forest Results:")
            print(f"Best parameters: {random_search.best_params_}")
            print(f"MSE: {mse}")
            print(f"R Squared: {r2}")

            feature_importances = best_rf.feature_importances_
            for feature, importance in zip(X.columns if hasattr(X, 'columns') else range(X.shape[1]), feature_importances):
                mlflow.log_metric(f"importance_{feature}", importance)
                print(f"{feature}: {importance}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

    return best_rf, mse, r2
def train_xgboost(X_train, X_test, y_train, y_test, data_type, run_name="XGBoost"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("data_preprocessing", data_type)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 5, 7]
        }
        
        xgb = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        
        y_pred = best_xgb.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # Calculate RMSE
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)  # Log RMSE
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(best_xgb, "xgboost_model")
        
        print("XGBoost Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")  # Print RMSE
        print(f"R Squared: {r2}")
        
        feature_importances = best_xgb.feature_importances_
        for feature, importance in zip(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]), feature_importances):
            mlflow.log_metric(f"importance_{feature}", importance)
            print(f"{feature}: {importance}")

    return best_xgb, mse, rmse, r2 


def train_random_forest_simple(X_train, X_test, y_train, y_test, data_type, run_name="Simple Random Forest"):
    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_param("model_type", "Simple Random Forest")
            mlflow.log_param("data_preprocessing", data_type)

            # Define the model with fixed hyperparameters
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                max_features=0.5,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )

            # Fit the model
            rf.fit(X_train, y_train)

            # Make predictions
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_params({
                "n_estimators": rf.n_estimators,
                "max_depth": rf.max_depth,
                "min_samples_split": rf.min_samples_split,
                "max_features": rf.max_features
            })
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(rf, "simple_random_forest_model")

            print("Simple Random Forest Results:")
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
            print(f"R Squared: {r2}")

            # Log feature importances
            feature_importances = rf.feature_importances_
            for feature, importance in zip(X.columns if hasattr(X, 'columns') else range(X.shape[1]), feature_importances):
                mlflow.log_metric(f"importance_{feature}", importance)
                print(f"{feature}: {importance}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

    return rf, mse, rmse, r2