from trainers.base_trainer import BaseTrainer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

class XGBoostTrainer(BaseTrainer):
    """
    Trainer para modelos de classificação com XGBoost.
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
        """Treina um modelo XGBClassifier."""
        if params is None:
            params = {}
        
        # O XGBoost pode usar o `early_stopping_rounds` se um `eval_set` for fornecido
        # Por simplicidade aqui, vamos apenas treinar com os parâmetros dados.
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Avalia o modelo XGBoost."""
        preds = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds)
        }
        return metrics
