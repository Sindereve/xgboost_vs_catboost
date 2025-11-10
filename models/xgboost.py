import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import xgboost as xgb

def test_xgb_with_recoder(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        params: dict,
        task_type: str
    ):
    """XGBoost с автоматическим ре-кодером"""
    try:
        start_time = time.time()
        
        # str в category
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        
        if task_type == 'clf':
            model = xgb.XGBClassifier(
                **params,
                enable_categorical=True,
            )
        else:
            model = xgb.XGBRegressor(
                **params,
                enable_categorical=True,
            )
        
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test)
        return model, preds, train_time
    except Exception as e:
        print(f"Error [{test_xgb_with_recoder.__name__}]: {e}")

def test_xgb_manual_encoding(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        params: dict,
        task_type: str
    ):
    """XGBoost с ручным кодированием"""
    try:
        start_time = time.time()
        
        # Разделяем числовые и категориальные признаки
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        numerical_columns = X_train.select_dtypes(exclude=['object']).columns
        
        # One-Hot Encoding и Ordinal в зависимости от количества уникальных значений
        low_cardinality = [col for col in categorical_columns if X_train[col].nunique() < 10]
        high_cardinality = [col for col in categorical_columns if X_train[col].nunique() >= 10]
        
        # Применяем кодирование
        if low_cardinality:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_train_ohe = ohe.fit_transform(X_train[low_cardinality])
            X_test_ohe = ohe.transform(X_test[low_cardinality])
        
        if high_cardinality:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_ord = oe.fit_transform(X_train[high_cardinality])
            X_test_ord = oe.transform(X_test[high_cardinality])
        
        # Собираем обратно
        X_train_encoded = np.hstack([X_train[numerical_columns].values] + 
                                ([X_train_ohe] if low_cardinality else []) + 
                                ([X_train_ord] if high_cardinality else []))
        
        X_test_encoded = np.hstack([X_test[numerical_columns].values] + 
                                ([X_test_ohe] if low_cardinality else []) + 
                                ([X_test_ord] if high_cardinality else []))
        
        if task_type == 'clf':
            model = xgb.XGBClassifier(
                **params
            )
        else:
            model = xgb.XGBRegressor(
                **params
            )
        
        model.fit(X_train_encoded, y_train)
        train_time = time.time() - start_time
        
        preds = model.predict(X_test_encoded)
        return model, preds, train_time
    except Exception as e:
        print(f"Error [{test_xgb_manual_encoding.__name__}]: {e}")