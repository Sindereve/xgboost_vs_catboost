import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import time


def test_catboost(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        params: dict,
        task_type: str ='clf',
    ):
    """CatBoost с автоматической обработкой категорий"""
    try:
        start_time = time.time()
        
        categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            X_train[col] = X_train[col].fillna('MISSING')
            X_test[col] = X_test[col].fillna('MISSING')
        
        if task_type == 'clf':
            model = CatBoostClassifier(
                **params,
                cat_features=categorical_columns
            )
        else:
            model = CatBoostRegressor(
                **params,
                cat_features=categorical_columns
            )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test)
        return model, preds, train_time
    except Exception as e:
        print(f"Error [{test_catboost.__name__}]: {e}")