from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xg
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

# REFERNCES
# DATASET LINK : https://www.kaggle.com/karangadiya/fifa19
# Sklearn & Xgboost Documentations
# https://www.kaggle.com/yousefmagdy/analyzing-and-cleaning-fifa19-data
# https://www.kaggle.com/aryantiwari123/fifa-19-complete-player-dataset


# CONVERT MONEY VALUES to INT VALUES -------------------------------
def text_to_num(value):

    zeros = {"K": 1000, "M": 1000000, "B": 1000000000}
    if not isinstance(value, str):
        return 0

    if value[-1] in zeros:
        num = value[:-1]
        multiplier = value[-1]
        result = int(float(num) * zeros[multiplier])
        return result
    else:
        return float(value)


# CONVERT HEIGHT VALUES to INT VALUES -------------------------------
def height_to_inch(height):
    lst = []
    h = 1
    if isinstance(height, str):
        lst = height.split("'")
        a = int(lst[0])
        b = int(lst[1])
        h = (a * 12) + b
    return h


# CONVERT YEARS to INT VALUES ---------------------------------------
def contract_years(years):
    if isinstance(years, str):
        temp = years[-4:]
        if temp != "0":
            return int(temp) - 2018
        else:
            return 0


# DATA LOADING, CLEANING AND FEATURE ENGINEERING ---------------------
def data_processing_cleaning():
    data = pd.read_csv("data/data.csv")
    player_name = data["Name"]
    data = data.drop(
        [
            "Photo",
            "Flag",
            "Club Logo",
            "Unnamed: 0",
            "Real Face",
            "ID",
            "Club Logo",
            "Jersey Number",
            "Name",
        ],
        axis=1,
    )

    # data = data[data["Value"] != "€0"]
    # data["Value"] = data["Value"].str.replace("€", "")
    data["Wage"] = data["Wage"].str.replace("€", "")
    data["Release Clause"] = data["Release Clause"].str.replace("€", "")

    data["Work Rate"] = data["Work Rate"].fillna("Medium/ Medium")

    lst = data["Work Rate"].str.split("/ ", n=1, expand=True)
    data["Work Rate 1"] = lst[0]
    data["Work Rate 2"] = lst[1]
    data["Work Rate 1"] = data["Work Rate 1"].replace(
        ["High", "Medium", "Low"], [1, 2, 3]
    )
    data["Work Rate 2"] = data["Work Rate 2"].replace(
        ["High", "Medium", "Low"], [1, 2, 3]
    )
    data = data.drop("Work Rate", axis=1)

    data["Body Type"] = data["Body Type"].replace(
        [
            "Neymar",
            "Messi",
            "C. Ronaldo",
            "Courtois",
            "Shaqiri",
            "Akinfenwa",
            "PLAYER_BODY_TYPE_25",
        ],
        [1, 2, 2, 2, 3, 3, 2],
    )
    data["Body Type"] = data["Body Type"].replace(
        ["Lean", "Normal", "Stocky"], [1, 2, 3]
    )
    data["Body Type"] = data["Body Type"].fillna(2)

    data["Position"] = data["Position"].fillna(0)
    data["Position"] = data["Position"].replace(
        [
            "GK",
            "RF",
            "ST",
            "LW",
            "RCM",
            "LF",
            "RS",
            "RCB",
            "LCM",
            "CB",
            "LDM",
            "CAM",
            "CDM",
            "LS",
            "LCB",
            "RM",
            "LAM",
            "LM",
            "LB",
            "RDM",
            "RW",
            "CM",
            "RB",
            "RAM",
            "CF",
            "RWB",
            "LWB",
        ],
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ],
    )

    # data['Position'] = data['Position'].fillna(0)
    data["Weight"] = data["Weight"].str.replace("lbs", "")
    data["Weight"] = data["Weight"].fillna(0)
    data["Weight"] = data["Weight"].astype(int)
    data["Weight"] = data["Weight"].replace(0, data["Weight"].mean())

    data["Loaned From"] = (
        data["Loaned From"].where(data["Loaned From"].isnull(), 1).fillna(0).astype(int)
    )

    data["Preferred Foot"] = data["Preferred Foot"].fillna(0)
    data["Preferred Foot"] = data["Preferred Foot"].replace(["Right", "Left"], [0, 1])

    data["Contract Valid Until"] = data["Contract Valid Until"].fillna(0).astype(str)
    lst = []
    for val in data["Contract Valid Until"]:
        lst.append(contract_years(val))

    data["Contract Years Left"] = lst
    data["Contract Years Left"] = data["Contract Years Left"].astype(int)
    data = data.drop("Contract Valid Until", axis=1)

    # lst = []
    # for val in data["Value"]:
    #    lst.append(text_to_num(val))
    # data["Value"] = lst

    lst = []
    for val in data["Wage"]:
        lst.append(text_to_num(val))
    data["Wage"] = lst

    lst = []
    for val in data["Release Clause"]:
        lst.append(text_to_num(val))
    data["Release Clause"] = lst
    data["Release Clause"] = data["Release Clause"].astype(int)

    lst = []
    for val in data["Height"]:
        lst.append(height_to_inch(val))

    data["Height"] = lst
    data["Height"] = data["Height"].replace(1, int(data["Height"].mean()))

    data["Joined"] = data["Joined"].replace(np.nan, 0)
    # 0 SIGNIGIES PLAYER IS ON LOAN
    lst = []
    for val in data["Joined"]:
        if val == 0:
            lst.append(0)
        elif val != 0:
            temp = val.split(",")
            lst.append(temp[-1])

    data["Joined"] = lst
    data["Joined"] = data["Joined"].astype(int)

    # GENERATES THE NATIONALITY PLOT
    Visualization.nationality_plot(data)
    nationalities = data["Nationality"].value_counts()
    tier1_nation = nationalities[(nationalities > 255)].index.tolist()
    lst = []
    for val in data["Nationality"]:
        if val in tier1_nation:
            lst.append(0)
        else:
            lst.append(1)
    data["Nationality"] = lst
    data["Nationality"] = data["Nationality"].astype(int)

    tier1_clubs = (
        (data.groupby("Club").agg({"Wage": "sum"}))
        .sort_values(by="Wage", ascending=False)
        .head(20)
    )
    lst = []
    for val in data["Club"]:
        if val in tier1_clubs.index.tolist():
            lst.append(0)
        else:
            lst.append(1)
    data["Club"] = lst
    data["Club"] = data["Club"].astype(int)

    attributes = [
        "LS",
        "ST",
        "RS",
        "LW",
        "LF",
        "CF",
        "RF",
        "RW",
        "LAM",
        "CAM",
        "RAM",
        "LM",
        "LCM",
        "CM",
        "RCM",
        "RM",
        "LWB",
        "LDM",
        "CDM",
        "RDM",
        "RWB",
        "LB",
        "LCB",
        "CB",
        "RCB",
        "RB",
    ]

    for column in attributes:
        lst = []
        for val in data[column]:
            if type(val) == str:
                temp = val.split("+")
                lst.append(int(temp[0]) + int(temp[1]))
            else:
                lst.append(val)
        data[column] = lst
        data[column] = data[column].fillna(data[column].mean())
        data[column] = data[column].astype(int)

    int_columns = [
        "International Reputation",
        "Weak Foot",
        "Skill Moves",
        "Crossing",
        "Finishing",
        "HeadingAccuracy",
        "ShortPassing",
        "Volleys",
        "Dribbling",
        "Curve",
        "FKAccuracy",
        "LongPassing",
        "BallControl",
        "Acceleration",
        "SprintSpeed",
        "Agility",
        "Reactions",
        "Balance",
        "ShotPower",
        "Jumping",
        "Stamina",
        "Strength",
        "LongShots",
        "Aggression",
        "Interceptions",
        "Positioning",
        "Vision",
        "Penalties",
        "Composure",
        "Marking",
        "StandingTackle",
        "SlidingTackle",
        "GKDiving",
        "GKHandling",
        "GKKicking",
        "GKPositioning",
        "GKReflexes",
    ]
    data[int_columns] = data[int_columns].fillna((data[int_columns].mean()).astype(int))

    # CLEANING LABEL (Y)
    data = clean_label(data)

    # PLOTTING METHODS CALLED TO GENERATE FIGURES WITHOUT EXPLICITLY DOING IN MAIN()
    Visualization.contract_value_plot(data)
    Visualization.release_value_plot(data)

    y = data["Value"]
    X = data.drop(["Value"], axis=1)

    return X, y, player_name


# CLEANS THE TARGET COLUMN AND FILLING IT WITH MODE OF ITS CLUSTER --
def clean_label(data):
    data["Value"] = data["Value"].str.replace("€0", "0")
    data["Value"] = data["Value"].str.replace("€", "")

    lst = []
    for val in data["Value"]:
        lst.append(text_to_num(val))
    data["Value"] = lst

    Visualization.y_val_plotting(data["Value"], "Y_log_values_before")
    missing_index = data.Value[data["Value"] == 0.0].index.tolist()

    missing_modes = []
    for i in data.Overall[data["Value"] == 0.0]:
        missing_modes.append(int(data.Value[data["Overall"] == i].mode()))

    for (i, j) in zip(missing_index, missing_modes):
        data["Value"].loc[i] = j

    Visualization.y_val_plotting(data["Value"], "Y_log_values_after")
    return data


def feature_selection(X, y):

    selection = SelectFromModel(Lasso(alpha=0.0001, random_state=42)).fit(
        MinMaxScaler().fit_transform(X), y
    )
    selected_features = X.columns[(selection.get_support())]

    Visualization.correlational_plot(X)

    return X[selected_features]


# FEATURE SELECTION USING RECURSIVE FEEEATURE ELIMINATION (NOT USED)-
def feature_selection_RFECV(X, y):

    rfecv = RFECV(Lasso(alpha=0.0001, random_state=42)).fit(X, y)

    Visualization.feature_selection_plot(rfecv)
    Visualization.correlational_plot(X)
    return X.loc[:, rfecv.get_support()]


# CROSS VALIDATION SCORE AND RMSE OF THE MODEL ---------------------
def cv_scores(X, y):
    SKF = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    SKF.get_n_splits(X, y)
    cv_score_test = []
    cv_score_train = []
    rmse = []
    for train_index, test_index in SKF.split(X, y):

        X_train, y_train, X_test, y_test = (
            X.iloc[train_index],
            y.iloc[train_index],
            X.iloc[test_index],
            y.iloc[test_index],
        )

        regressor = xg.XGBRegressor(
            objective="reg:squarederror",
            min_child_weight=1,
            booster="gbtree",
            n_estimators=200,
            seed=123,
            learning_rate=0.1,
        )

        model = Pipeline(
            (
                ("MinMaxScaler", MinMaxScaler()),
                ("Polynomial", PolynomialFeatures(degree=2)),
                ("XGBoost", regressor),
            )
        )

        model.fit(X_train, y_train)

        cv_score_test.append(model.score(X_test, y_test))
        cv_score_train.append(model.score(X_train, y_train))
        y_predict = model.predict(X_test)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_predict)))

    print(
        "CV score for train set is:",
        np.round(np.mean(cv_score_train), 3),
        "CV score for test set is:",
        np.round(np.mean(cv_score_test), 3),
        "CV score RMSE for test set is:",
        np.round(np.mean(rmse), 3),
    )


# INITILIZES and TRAIN the MODEL and PREDICT VALUE ------------------
def modeling(X, y, player_name):

    X = X.assign(Name=player_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    player_name_Xtrain = X_train["Name"]
    player_name_Xtest = X_test["Name"]
    X_test = X_test.drop(["Name"], axis=1)
    X_train = X_train.drop(["Name"], axis=1)

    regressor = xg.XGBRegressor(
        objective="reg:squarederror",
        min_child_weight=1,
        booster="gbtree",
        n_estimators=200,
        seed=123,
        learning_rate=0.1,
    )

    model = Pipeline(
        (
            ("MinMaxScaler", MinMaxScaler()),
            ("Polynomial", PolynomialFeatures(degree=2)),
            ("XGBoost", regressor),
        )
    )

    model.fit(X_train, y_train)
    print("Train Data Score:", model.score(X_train, y_train))
    y_train_predicted = model.predict(X_train)
    my_scores_train = scoring(y_train, y_train_predicted, np.sqrt(len(X_train.axes[0])))

    y_test_predicted = model.predict(X_test)
    print("Train Data Score:", model.score(X_test, y_test))
    my_scores_test = scoring(y_test, y_test_predicted, np.sqrt(len(X_test.axes[0])))

    scores_df = pd.DataFrame(
        columns=[
            "Average",
            "R square",
            "Std Deviation",
            "RMSE",
            "Std Error",
            "Evaluation Variance Score",
        ],
        index=["Training Data", "Testing Data"],
    )
    scores_df.loc["Training Data"] = pd.Series(
        {
            "Average": my_scores_train[0],
            "R square": my_scores_train[1],
            "Std Deviation": my_scores_train[2],
            "RMSE": my_scores_train[3],
            "Std Error": my_scores_train[4],
            "Evaluation Variance Score": my_scores_train[5],
        }
    )
    scores_df.loc["Testing Data"] = pd.Series(
        {
            "Average": my_scores_test[0],
            "R square": my_scores_test[1],
            "Std Deviation": my_scores_test[2],
            "RMSE": my_scores_test[3],
            "Std Error": my_scores_test[4],
            "Evaluation Variance Score": my_scores_test[5],
        }
    )

    print("Scores for both the training and testing sets are:")
    print("--------------------------------------------------")
    print(scores_df)

    predicted_dataframe_train = pd.concat([player_name_Xtrain, y_train], axis=1)
    predicted_dataframe_train["Predicted Value"] = (y_train_predicted).astype(int)
    export_to_csv(predicted_dataframe_train, "outputs/predicted_training_set.csv")

    predicted_dataframe = pd.concat([player_name_Xtest, y_test], axis=1)
    predicted_dataframe["Predicted Value"] = (y_test_predicted).astype(int)
    export_to_csv(predicted_dataframe, "outputs/predicted_testing_set.csv")

    export_to_csv(X_test, "data/X_test.csv")
    export_to_csv(X_train, "data/X_train.csv")

    export_to_csv(scores_df, "outputs/scores.csv")


# HYPERPARAMETER TUNING for XGBRegressor MODEL     ------------------
def hyperparameter_tuning(X_train, y_train):

    parameters = {
        "n_estimators": [100, 200, 300, 400, 500, 800, 1200],
        "max_depth": [2, 3, 4, 5],
        "booster": ["gblinear", "gbtree"],
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25],
        "min_child_weight": [1, 2, 3, 4],
    }

    random_search_cv = RandomizedSearchCV(
        estimator=xg.XGBRegressor(),
        param_distributions=parameters,
        cv=5,
        n_iter=40,
        scoring="neg_mean_squared_error",
        verbose=5,
        return_train_score=True,
        random_state=42,
    ).fit(X_train, y_train)

    print("BEST MODEL PARAMETERS:", random_search_cv.best_estimator_)

    return random_search_cv.best_estimator_


# GIVES DIFFERENT SCORES FOR BOTH TRAINING AND TESTING SET ------------
def scoring(y, y_predict, n_sample_root):
    avg = np.mean(y - y_predict)
    r2 = r2_score(y, y_predict)
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    std = np.std(y - y_predict)
    std_err = std / n_sample_root
    evs = explained_variance_score(y, y_predict)

    return avg, r2, std, rmse, std_err, evs


# EXPORT FILE TO CSV --------------------------------------------------
def export_to_csv(data, filename):
    data.to_csv(filename, index=False)
    print("file successfully exported to :", filename)


# Class Containing Static Methods for all the visualization methods used
class Visualization:

    # ------------y_val_plotting METHOD  -----------------------
    # PARAMETERS: Y and Name of the file  ----------------------
    # RETURNS   : Saves a figure  ------------------------------

    # REFERENCE : SNS documentation
    @staticmethod
    def y_val_plotting(y, name) -> None:
        yplotting = sns.displot(np.log1p(y), kde=False, height=16, aspect=1)
        yplotting.fig.suptitle(name)
        filename = "figs/" + name + ".png"
        plt.savefig(filename)

    # ------------correlational_plot METHOD  -------------------
    # PARAMETERS: X                       ----------------------
    # RETURNS   : Saves a figure  ------------------------------

    # REFERENCE : SNS & MATPLOTLIB documentation
    @staticmethod
    def correlational_plot(X):

        plt.matshow(X.corr(), fignum=(plt.figure(figsize=(40, 40))).number)

        plt.xticks(range(X.shape[1]), X.columns, fontsize=18, rotation=30)
        plt.yticks(range(X.shape[1]), X.columns, fontsize=18)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=18)
        plt.savefig("figs/correlational_matrix.png")

    # ------------feature_selection_plot METHOD  -------------------
    # PARAMETERS: Feature selection Object    ----------------------
    # RETURNS   : Saves a figure      ------------------------------

    # REFERENCE : SNS & MATPLOTLIB documentation
    @staticmethod
    def feature_selection_plot(rfecv):
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker="x")
        plt.title(f"RFE Features Selected: {rfecv.n_features_}")
        plt.xlabel("No of Features")
        plt.ylabel("Score")
        plt.savefig("figs/feature_selection.png")

    # ------------contract_value_plot METHOD     -------------------
    # PARAMETERS: DataFrame                   ----------------------
    # RETURNS   : Saves a figure      ------------------------------

    # REFERENCE : SNS & MATPLOTLIB documentation
    @staticmethod
    def contract_value_plot(data):
        sns.lmplot(
            x="Value",
            y="Contract Years Left",
            data=data,
            markers="v",
            scatter_kws={"color": "blue"},
            line_kws={"linewidth": 3, "color": "red"},
            aspect=2,
        )
        plt.title("Figure : \n\n Contract Years Left vs Value")
        plt.savefig("figs/contract_value.png", bbox_inches="tight")

    # ------------release_value_plot METHOD      -------------------
    # PARAMETERS: Dataframe                   ----------------------
    # RETURNS   : Saves a figure      ------------------------------

    # REFERENCE : SNS & MATPLOTLIB documentation
    @staticmethod
    def release_value_plot(data):
        sns.lmplot(
            x="Value",
            y="Release Clause",
            data=data,
            markers="v",
            scatter_kws={"color": "blue"},
            line_kws={"linewidth": 3, "color": "red"},
            aspect=2,
        )
        plt.title("Figure : \n\n Release Clause vs Value")
        plt.savefig("figs/release_value.png", bbox_inches="tight")

    # ------------nationality_plot METHOD      -------------------
    # PARAMETERS: Dataframe                   ----------------------
    # RETURNS   : Saves 2 figures      ------------------------------

    # REFERENCE : SNS & MATPLOTLIB documentation
    @staticmethod
    def nationality_plot(data):

        nationalities = data["Nationality"].value_counts().reset_index()
        nationalities.columns = ["Nations", "Player_Counts"]

        nationalities_more = nationalities.loc[nationalities["Player_Counts"] > 255]
        nationalities_less = nationalities.loc[nationalities["Player_Counts"] < 255]

        sns.catplot(
            y="Nations",
            x="Player_Counts",
            data=nationalities_more,
            palette="colorblind",
            height=15,
            kind="bar",
            aspect=2,
        )
        plt.title(
            f"Nation Wise Players Counts only {nationalities_more.shape[0]} countries have significant players",
            fontsize=25,
        )
        plt.savefig("figs/more_players_nations.png", bbox_inches="tight")

        sns.catplot(
            y="Nations",
            x="Player_Counts",
            data=nationalities_less,
            palette="colorblind",
            height=15,
            kind="bar",
            aspect=1,
        )
        plt.title(
            f"Nation Wise Players Counts: {nationalities_less.shape[0]} countries have less players",
            fontsize=25,
        )
        plt.savefig("figs/less_players_nations.png", bbox_inches="tight")


# -------------------------SORT AUC METHOD-----------------------------
# PARAMETERS
# auc_scores : Conatains list of auc scores                  ----------
# column_names : Conatains list of features name(Column name) ---------
# RETURNS : A Dataframe containing sorted(further from 0.5) AUC values.

# REFERENCE: MY CODE ASSIGNMENT 3...          -------------------------
def sort_auc_scores(auc_scores, column_names) -> None:
    temp = []

    for value in auc_scores:
        if value < 0.5:
            value = 1 - value
        temp.append(value)

    # A DF with a column AUC_adjusted which contains adjusted AUC values.
    # Eg. AUC score of 0.3 is stated as 1 - 0.3 = 0.7 (Furtherest from 0.5)
    # The column is present only for sorting and doesnt change the AUC.

    return (
        pd.DataFrame({"Features": column_names, "AUC": auc_scores, "AUC_new": temp})
        .sort_values(by=["AUC_new"], ascending=False)
        .drop(columns="AUC_new")
    )


# Same as Assignment 1, 2 & 3 with regression dataset -----------------
def question2(X_train, y_train) -> None:

    auc_features = []

    y = y_train.apply(lambda x: 0 if x <= y_train.mean() else 1)

    for columnName in X_train.columns:

        auc_features.append(roc_auc_score(y, X_train[columnName]))

    sorted_auc = sort_auc_scores(auc_features, X_train.columns)
    sorted_auc.to_json(Path(__file__).resolve().parent / "outputs/aucs.json")

    print("---- Rounded AUC Scores : All the Features/Columns ----")
    print(sorted_auc.round(3))

    print("---- Rounded AUC scores : Top 10 Features/Columns ----")
    print(sorted_auc.round(3).head(10))

    print("Question-2 Successfully Executed")


if __name__ == "__main__":

    # VISUALIZATION ARE DONE IN VARIOUS LINES OF THE CODE
    # ALL THE IMAGES ARE EXPORTED TO figs/..
    # X_train and X_test ARE EXPORTED TO data/..
    # SCORING FILE, TRAINING AND TESTING PREDICTED FILES ARE EXPORTED TO outputs/..

    # DATA LOADING
    X, y, player_names = data_processing_cleaning()

    # FEATURE SELECTION
    X_selected = feature_selection(X, y)

    # AUC SCORES
    question2(X_selected, y)

    # APPLYING MODEL
    # CV SCORES ARE PRINTED
    cv_scores(X_selected, y)
    modeling(X_selected, y, player_names)
