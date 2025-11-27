House Price Prediction ML + Deployment

This project predicts house sale prices using the Ames housing dataset from Kaggle.  
The project demonstrates a complete machine-learning workflow including data exploration, preprocessing, training regression models, evaluating performance, tuning hyperparameters, saving the best model, and deploying a working interface using Streamlit.

Dataset Details
- Source: Kaggle House Prices - Advanced Regression Techniques
- Train samples: 1460
- Test samples: 1459
- Features: 79 columns describing homes
- Target column: SalePrice (the actual sale price of a house)
- Geography: Ames, Iowa, USA
- Feature categories include:
  - Property size (LotArea, GrLivArea, TotalBsmtSF, 1stFlrSF, GarageArea)
  - Quality ratings (OverallQual, OverallCond, KitchenQual, GarageFinish)
  - Construction years (YearBuilt, YearRemodAdd, GarageYrBlt, YrSold)
  - Rooms and utilities (FullBath, BedroomAbvGr, TotRmsAbvGrd, Fireplaces, PoolArea)
  - Building structure (Neighborhood, HouseStyle, Heating, RoofStyle, Exterior1st)
  - Additional realistic home attributes such as basement condition, garage finish, porch sizes, fence type, sale condition and materials.

ML Training Pipeline (Phase 1)
The machine-learning pipeline was built in Kaggle Notebook. Steps completed:
1. Loaded the dataset using pandas
2. Checked and inspected missing values
3. Filled missing numeric values using median strategy and scaled them using StandardScaler
4. Filled missing categorical values using most frequent strategy and encoded them using OneHotEncoder
5. Split the data using train_test_split to create a training and validation set
6. Trained regression models:
   - Linear Regression for baseline comparison
   - Random Forest Regressor for non-linear learning and feature interaction
   - XGBoost Regressor as an additional high-performance model for experiment
7. Evaluated models using RMSE, MAE and R² score
8. Tuned the Random Forest model using RandomizedSearchCV
9. Selected 6 important features for deployment model:
OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt
10. Saved the final trained model using joblib
11. Generated submission.csv for Kaggle competition format

Deployment (Phase 2)
A simplified 6-feature Random Forest model was deployed using Streamlit for real-time predictions in the browser.

To run the app locally, open a terminal inside the project folder and run:

```cmd
python -m streamlit run app.py

