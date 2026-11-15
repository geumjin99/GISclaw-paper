"""Reference solution for task 28: subregion interaction strength with a GridSearchCV-tuned
Random Forest.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42


# top-level script so that all intermediate tables stay in the namespace
socio = pd.read_csv("dataset/socioeconomic_data.csv")
od = pd.read_csv("dataset/od_data.csv")

# 1. aggregate OD records to one interaction per (origin, destination) subregion pair
interactions = (od.groupby(["Origin", "Destination"], as_index=False)["FlowVolume"].sum()
                  .rename(columns={"FlowVolume": "TotalFlow"}))
print(f"{len(od)} OD records -> {len(interactions)} subregion pairs "
      f"({(interactions.Origin == interactions.Destination).sum()} intra-subregion)")

# 2. attach origin and destination socio-economic attributes
o = socio.add_suffix("_O").rename(columns={"Subregion_O": "Origin"})
d = socio.add_suffix("_D").rename(columns={"Subregion_D": "Destination"})
merged = interactions.merge(o, on="Origin", how="inner").merge(d, on="Destination", how="inner")
assert len(merged) == len(interactions)
features = [c for c in merged.columns if c.endswith(("_O", "_D"))]
X, y = merged[features], merged["TotalFlow"]

# 3. train/test split, 4. GridSearchCV (5-fold, MSE), 5. evaluate the best model on test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
grid = {"n_estimators": [100, 300], "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10], "min_samples_leaf": [1, 4]}
search = GridSearchCV(RandomForestRegressor(random_state=SEED), grid,
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error", n_jobs=4)
search.fit(X_train, y_train)
best = search.best_estimator_
y_pred = best.predict(X_test)
test_mse = float(mean_squared_error(y_test, y_pred))
test_r2 = float(r2_score(y_test, y_pred))
baseline_mse = float(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print("best params:", search.best_params_)
print(f"CV MSE {-search.best_score_:.1f}; test MSE {test_mse:.1f} (R2 {test_r2:.3f}); "
      f"mean-only baseline MSE {baseline_mse:.1f}")

pred_table = merged.loc[X_test.index, ["Origin", "Destination", "TotalFlow"]].copy()
pred_table["PredictedFlow"] = y_pred
pred_table.to_csv("pred_results/rf_test_predictions.csv", index=False)
merged.to_csv("pred_results/subregion_interactions.csv", index=False)
importance = pd.Series(best.feature_importances_, index=features).sort_values(ascending=False)
json.dump({"best_params": search.best_params_, "cv_mse": -search.best_score_, "test_mse": test_mse,
           "test_r2": test_r2, "baseline_mse": baseline_mse, "n_pairs": len(merged),
           "feature_importance": importance.round(4).to_dict()},
          open("pred_results/rf_model_performance.json", "w"), indent=1)

