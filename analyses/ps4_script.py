#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glum import TweedieDistribution
from lightgbm import LGBMRegressor, plot_metric
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ps3.data import create_sample_split, load_transform
from ps3.evaluation import evaluate_metrics

# %%
# Load and prepare data
df = load_transform()

weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]

df = create_sample_split(df, "IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categorical_cols = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
numeric_cols = ["BonusMalus", "Density"]

predictors = categorical_cols + numeric_cols
X_train_t = df[predictors].iloc[train]
X_test_t = df[predictors].iloc[test]

y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]


# %%
# Exercise 1
# Plots of average claims by BonusMalus
df_train["BonusMalusGroup"] = pd.cut(df_train["BonusMalus"], 10)
df_train.groupby("BonusMalusGroup")["PurePremium"].mean().plot(kind="bar")
plt.show()

# The graphs show that average claim size generally increases with BonusMalus
# but there are some data points with high BonusMalus that have decreasing claim size.
# If a monotonicity constraint is included, these points will be forced to predict
# a higher claim size (all else equal), which is more in line with economic intuition.

# %%
# Define preprocessor and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categorical_cols),
    ]
)
preprocessor.set_output(transform="pandas")

# Defines monotone constraints for the column relating to BonusMalus
preprocessor.fit(df_train[predictors])
feature_names = preprocessor.get_feature_names_out()
monotone_constraints = [
    1 if "BonusMalus" in f else 0
    for f in feature_names
]

constrained_lgb = LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power = 1.5,
    monotone_constraints = monotone_constraints
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', constrained_lgb)
])


TweedieDist = TweedieDistribution(1.5)

# def tweedie_scorer(y_true, y_pred, sample_weight=None):
#     if sample_weight is None:
#         sample_weight = np.ones_like(y_true)
#     return TweedieDist.deviance(y_true, y_pred, sample_weight=sample_weight) / np.sum(sample_weight)

# tweedie_sklearn_scorer = make_scorer(
#     tweedie_scorer,
#     greater_is_better=False
# )


# %%
# Run cross-validation to find the best learning rate and number of estimators
param_grid = {
    'regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__n_estimators': [100, 150, 200]
}

cv = GridSearchCV(
    model_pipeline,
    param_grid,
    cv = KFold(n_splits = 5, shuffle = True, random_state=42)
)

cv.fit(
    X_train_t, y_train_t,
    regressor__sample_weight=w_train_t
)

df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)

TweedieDist = TweedieDistribution(1.5)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%
# Exercise 2:
# Train model with best parameters
best_params = cv.best_params_
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', constrained_lgb)
])
model_pipeline.set_params(**best_params)

preprocessor.fit(X_train_t)
X_train_transformed = preprocessor.transform(X_train_t)
X_test_transformed = preprocessor.transform(X_test_t)

model_pipeline.named_steps['regressor'].fit(
    X_train_transformed, y_train_t,
    eval_set=[(X_train_transformed, y_train_t), (X_test_transformed, y_test_t)],
    eval_names=['train', 'test'],
    sample_weight=w_train_t
)

# %%
# Plot the evolution of the score
regressor = model_pipeline.named_steps['regressor']
ax = plot_metric(regressor)
plt.show()

# The graph shows test loss decreasing until a plateau and not increasing,
# suggesting that no overfitting is happening


# %%
# Exercise 3
# Run the unconstrained lgbm model again
best_params = cv.best_params_
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', constrained_lgb)
])
model_pipeline.set_params(**best_params)
model_pipeline.set_params(regressor__monotone_constraints = '')

model_pipeline.fit(X_train_t, y_train_t, regressor__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)

#%%
# Evaluate predictions
df1 = evaluate_metrics(y_train_t, df_train["pp_t_lgbm_constrained"], w_train_t)
df2 = evaluate_metrics(y_train_t, df_train["pp_t_lgbm"], w_train_t)
df_metrics = pd.concat([df1, df2], axis=1)
df_metrics.columns = ["LGBM Constrained", "LGBM Unconstrained"]
df_metrics

# The unconstrained LGBM model performs better than the constrained one
# These were set to have the same other hyperparameters

