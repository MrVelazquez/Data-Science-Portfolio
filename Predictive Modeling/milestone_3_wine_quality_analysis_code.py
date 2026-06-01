# =============================================================================
# Milestone 3
# Adrian Velazquez
# DSC 680 Applied Data Science
#
# Project: Predicting Red and White Wine Quality Using Physicochemical Properties
# and Machine Learning
#
# Data Source: UCI Machine Learning Repository - Wine Quality Dataset
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
RED_FILE = "winequality-red.csv"
WHITE_FILE = "winequality-white.csv"


def load_wine_data():
    """Load red and white wine data from local files or directly from UCI."""
    red_source = RED_FILE if os.path.exists(RED_FILE) else RED_URL
    white_source = WHITE_FILE if os.path.exists(WHITE_FILE) else WHITE_URL
    red_df = pd.read_csv(red_source, sep=";")
    white_df = pd.read_csv(white_source, sep=";")
    red_df["wine_type"] = "red"
    white_df["wine_type"] = "white"
    wine_df = pd.concat([red_df, white_df], ignore_index=True)
    wine_df["quality_group"] = pd.cut(
        wine_df["quality"],
        bins=[0, 5, 6, 10],
        labels=["low", "medium", "high"],
        include_lowest=True
    )
    return red_df, white_df, wine_df


def create_summary_tables(wine_df):
    """Create core EDA summary tables."""
    missing_summary_df = wine_df.isna().sum().reset_index()
    missing_summary_df.columns = ["variable", "missing_count"]
    missing_summary_df["missing_percent"] = (
        missing_summary_df["missing_count"] / len(wine_df) * 100
    ).round(2)

    quality_dist_df = wine_df["quality"].value_counts().sort_index().reset_index()
    quality_dist_df.columns = ["quality", "count"]
    quality_dist_df["percent"] = (
        quality_dist_df["count"] / len(wine_df) * 100
    ).round(2)

    wine_type_summary_df = wine_df.groupby("wine_type").agg(
        observations=("quality", "count"),
        avg_quality=("quality", "mean"),
        avg_alcohol=("alcohol", "mean"),
        avg_volatile_acidity=("volatile acidity", "mean"),
        avg_residual_sugar=("residual sugar", "mean")
    ).round(3).reset_index()

    quality_group_means_df = wine_df.groupby("quality_group", observed=False)[
        ["alcohol", "volatile acidity", "density", "chlorides", "sulphates"]
    ].mean().round(3).reset_index()

    return missing_summary_df, quality_dist_df, wine_type_summary_df, quality_group_means_df


def run_separate_linear_regression_models(wine_df):
    """Run separate red and white wine linear regression baseline models."""
    baseline_rows = []
    correlation_rows = []

    for wine_label in tqdm(["red", "white"], desc="Running baseline models"):
        model_df = wine_df[wine_df["wine_type"] == wine_label].drop(
            columns=["wine_type", "quality_group"]
        )
        y_vals = model_df["quality"]
        x_vals = model_df.drop(columns=["quality"])
        x_train, x_test, y_train, y_test = train_test_split(
            x_vals,
            y_vals,
            test_size=0.25,
            random_state=3
        )

        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        train_pred = regressor.predict(x_train)
        test_pred = regressor.predict(x_test)

        baseline_rows.append({
            "wine_type": wine_label,
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "test_r2": r2_score(y_test, test_pred)
        })

        corr_series = model_df.corr(numeric_only=True)["quality"].drop("quality")
        corr_series = corr_series.sort_values(key=lambda values: values.abs(), ascending=False)
        for feature_name, corr_value in corr_series.head(5).items():
            correlation_rows.append({
                "wine_type": wine_label,
                "feature": feature_name,
                "correlation_with_quality": corr_value
            })

    baseline_results_df = pd.DataFrame(baseline_rows).round(3)
    separate_correlation_df = pd.DataFrame(correlation_rows).round(3)
    return baseline_results_df, separate_correlation_df


def run_random_forest_models(wine_df):
    """Run Random Forest classification and regression models."""
    feature_columns = [
        col_name for col_name in wine_df.columns
        if col_name not in ["quality", "quality_group"]
    ]
    x_vals = wine_df[feature_columns]
    y_quality = wine_df["quality"]
    y_group = wine_df["quality_group"].astype(str)
    numeric_features = [col_name for col_name in feature_columns if col_name != "wine_type"]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), ["wine_type"])
    ])

    x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
        x_vals,
        y_group,
        test_size=0.25,
        random_state=42,
        stratify=y_group
    )

    classifier = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    classifier.fit(x_train_class, y_train_class)
    class_predictions = classifier.predict(x_test_class)

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x_vals,
        y_quality,
        test_size=0.25,
        random_state=42
    )

    regressor = Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(
            n_estimators=250,
            random_state=42
        ))
    ])
    regressor.fit(x_train_reg, y_train_reg)
    reg_predictions = regressor.predict(x_test_reg)

    model_metrics_df = pd.DataFrame([{
        "classification_accuracy": round(accuracy_score(y_test_class, class_predictions), 3),
        "classification_weighted_f1": round(f1_score(y_test_class, class_predictions, average="weighted"), 3),
        "regression_rmse": round(np.sqrt(mean_squared_error(y_test_reg, reg_predictions)), 3),
        "regression_r2": round(r2_score(y_test_reg, reg_predictions), 3)
    }])

    encoded_feature_names = numeric_features + ["wine_type_white"]
    feature_importance_df = pd.DataFrame({
        "feature": encoded_feature_names,
        "importance": classifier.named_steps["rf"].feature_importances_
    }).sort_values("importance", ascending=False)

    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_test_class, class_predictions, labels=["low", "medium", "high"]),
        index=["actual_low", "actual_medium", "actual_high"],
        columns=["pred_low", "pred_medium", "pred_high"]
    )

    prediction_results_df = pd.DataFrame({
        "actual_quality": y_test_reg,
        "predicted_quality": reg_predictions,
        "residual": y_test_reg - reg_predictions
    })

    class_report_text = classification_report(y_test_class, class_predictions)
    return model_metrics_df, feature_importance_df, confusion_matrix_df, prediction_results_df, class_report_text


def create_final_visuals(wine_df, prediction_results_df):
    """Create final advanced visualizations for the white paper."""
    figure_files = []
    color_map = {
        "red": "#7B2D26",
        "white": "#C19A6B",
        "low": "#B56576",
        "medium": "#E6B980",
        "high": "#6A994E"
    }

    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=wine_df,
        x="alcohol",
        hue="quality_group",
        fill=True,
        common_norm=False,
        alpha=0.45,
        palette=[color_map["low"], color_map["medium"], color_map["high"]]
    )
    plt.title("Advanced Figure 1. Alcohol Distribution by Quality Group")
    plt.xlabel("Alcohol")
    plt.ylabel("Density")
    plt.tight_layout()
    fig_name = "advanced_figure_1_alcohol_distribution.png"
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    figure_files.append(fig_name)
    plt.show()

    pca_features = wine_df.drop(columns=["quality", "quality_group", "wine_type"])
    pca_scaled = StandardScaler().fit_transform(pca_features)
    pca_result = PCA(n_components=2, random_state=42).fit_transform(pca_scaled)
    pca_plot_df = wine_df[["quality_group", "wine_type"]].copy()
    pca_plot_df["pc1"] = pca_result[:, 0]
    pca_plot_df["pc2"] = pca_result[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_plot_df,
        x="pc1",
        y="pc2",
        hue="quality_group",
        style="wine_type",
        alpha=0.55,
        s=28,
        palette=[color_map["low"], color_map["medium"], color_map["high"]]
    )
    plt.title("Advanced Figure 2. PCA Map of Wine Chemistry Profiles")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    fig_name = "advanced_figure_2_pca_map.png"
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    figure_files.append(fig_name)
    plt.show()

    corr_df = wine_df.drop(columns=["quality_group", "wine_type"]).corr(numeric_only=True)
    mask_vals = np.triu(np.ones_like(corr_df, dtype=bool))
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_df,
        mask=mask_vals,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.7}
    )
    plt.title("Advanced Figure 3. Triangular Correlation Heatmap")
    plt.tight_layout()
    fig_name = "advanced_figure_3_correlation_heatmap.png"
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    figure_files.append(fig_name)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=prediction_results_df,
        x="actual_quality",
        y="predicted_quality",
        hue="residual",
        palette="coolwarm",
        alpha=0.55
    )
    plt.plot([3, 9], [3, 9], linestyle="--", color="black", linewidth=1)
    plt.title("Advanced Figure 4. Actual vs Predicted Quality")
    plt.xlabel("Actual Expert Quality Score")
    plt.ylabel("Predicted Quality Score")
    plt.tight_layout()
    fig_name = "advanced_figure_4_actual_vs_predicted.png"
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    figure_files.append(fig_name)
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxenplot(
        data=wine_df,
        x="quality",
        y="alcohol",
        hue="wine_type",
        palette={"red": color_map["red"], "white": color_map["white"]}
    )
    plt.title("Advanced Figure 5. Alcohol Levels Across Quality Scores")
    plt.xlabel("Expert Quality Score")
    plt.ylabel("Alcohol")
    plt.tight_layout()
    fig_name = "advanced_figure_5_alcohol_by_quality.png"
    plt.savefig(fig_name, dpi=200, bbox_inches="tight")
    figure_files.append(fig_name)
    plt.show()

    return figure_files


def main():
    """Run the full Milestone 3 final analysis."""
    red_df, white_df, wine_df = load_wine_data()
    missing_summary_df, quality_dist_df, wine_type_summary_df, quality_group_means_df = create_summary_tables(wine_df)
    baseline_results_df, separate_correlation_df = run_separate_linear_regression_models(wine_df)
    model_metrics_df, feature_importance_df, confusion_matrix_df, prediction_results_df, class_report_text = run_random_forest_models(wine_df)

    print("Combined dataset shape")
    print(wine_df.shape)
    print("Missing values total")
    print(int(wine_df.isna().sum().sum()))
    print("Duplicate rows")
    print(int(wine_df.duplicated().sum()))
    print("Quality distribution")
    print(quality_dist_df)
    print("Wine type summary")
    print(wine_type_summary_df)
    print("Quality group means")
    print(quality_group_means_df)
    print("Linear regression baseline results")
    print(baseline_results_df)
    print("Separate red and white wine correlations")
    print(separate_correlation_df)
    print("Random Forest model metrics")
    print(model_metrics_df)
    print("Top Random Forest feature importance")
    print(feature_importance_df.head(10))
    print("Classification confusion matrix")
    print(confusion_matrix_df)
    print("Classification report")
    print(class_report_text)

    figure_files = create_final_visuals(wine_df, prediction_results_df)
    print("Saved figure files")
    for fig_file in figure_files:
        print(fig_file)


if __name__ == "__main__":
    main()
