import pandas as pd
from sklearn.metrics import(
    r2_score, root_mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, recall_score, precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def calculation_metrics(
        y_pred,
        y_test,
        task: str
    ):
    try:
        if task == "clf":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            metrics = {
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
            }
        else:
            r2 = r2_score(y_pred, y_test)
            rmse = root_mean_squared_error(y_pred, y_test)
            mae = mean_absolute_error(y_pred, y_test)
            metrics = {
                "r2": round(r2, 4),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2)
            }
        return metrics
    except Exception as e:
        print(f"Error [{calculation_metrics.__name__}]: {e}")


def calculation_info_data(X: pd.DataFrame):
    try:
        count_row = X.shape[0]
        count_columns = X.shape[1]

        numeric_columns = list(X.select_dtypes('number'))
        count_num_columns = len(numeric_columns)

        cat_columns = list(X.select_dtypes(['object', 'bool']))
        count_cat_columns = len(cat_columns)

        if count_columns == cat_columns+numeric_columns:
            raise ValueError("Не верно расчитывается количество категориальных и числовых значений.")

        portion_cat_columns = count_cat_columns/count_columns
        portion_num_columns = count_num_columns/count_columns

        return {
            "data_count_columns": count_columns,
            "data_count_row": count_row,
            "data_count_num_columns": count_num_columns,
            "data_count_cat_columns": count_cat_columns,
            "data_portion_num_columns": round(portion_num_columns, 3),
            "data_portion_cat_columns": round(portion_cat_columns, 3),
        }
    except Exception as e:
        print(f"Error [{calculation_info_data.__name__}]: {e}")


def load_data(info: dict, random_state: int = 42):
    df = pd.read_csv(info["file"], low_memory=False)
    df = df.drop(info["trash_columns"], axis=1)
    
    X = df.drop(info["target"], axis=1)
    y = df[info["target"]]
    
    if info["task"] == "clf":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y = pd.Series(data=y_encoded, index=y.index, name=y.name)

    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def count_unique_data(df):
    """Быстрый подсчет уникальных значений во всем DataFrame"""
    unique_info = {}
    for col in df.columns:
        unique_info[col] = {
            'n_unique': df[col].nunique(),
            'dtype': str(df[col].dtype)
        }
    return unique_info