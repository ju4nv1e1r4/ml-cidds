from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)


class Optimize:
    def __init__(self, model, param_grid, X_train, X_test, y_train, y_test, cv, verbose=1, n_iter=25, scoring="recall", thresholds=None, ):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.verbose = verbose
        self.n_iter = n_iter
        self.scoring = scoring
        self.thresholds = thresholds or {
            "recall": 0.8,
            "precision": 0.8,
            "f1_score": 0.8,
            "accuracy": 0.8,
            "roc_auc": 0.8
        }
    
    def refine_param_grid(self, best_params, spread=0.2):
        """
        Cria um novo grid em torno dos melhores parâmetros encontrados.
        Ideal para uso no BayesSearchCV após RandomSearchCV.
        """
        new_grid = {}

        for param, best_val in best_params.items():
            if isinstance(best_val, (int, float)):
                min_val = best_val * (1 - spread)
                max_val = best_val * (1 + spread)

                if isinstance(best_val, int):
                    new_grid[param] = list(range(max(int(min_val), 1), int(max_val) + 1))
                else:
                    new_grid[param] = (min_val, max_val)  # formato contínuo para bayessearchcv
            else:
                # parâmetro categórico
                new_grid[param] = [best_val]

        return new_grid

    def with_random_search(self) -> dict:
        rs = RandomizedSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            n_jobs=2,
            verbose=self.verbose,
            n_iter=self.n_iter,
            scoring=self.scoring
        )

        rs.fit(
           self.X_train,
           self.y_train
        )

        best_model = rs.best_estimator_
        predictions = best_model.predict(self.X_test)
        proba = rs.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        cf = classification_report(self.y_test, predictions)
        rocauc_score = roc_auc_score(predictions, proba)
        precision = precision_score(self.y_test, predictions)
        cm = confusion_matrix(self.y_test, predictions)

        print(f"Best params: {rs.best_params_}\n")
        print("\nClassification Report\n", cf)
        print(f"Accuracy Score: {accuracy}\n")
        print(f"Recall Score: {recall}\n")
        print(f"Precision Score: {precision}\n")
        print(f"\nROC AUC Score: {rocauc_score}\n")
        print(f"Confusion Matrix: {cm}")

        best_params = rs.best_params_
        self.param_grid = self.refine_param_grid(best_params)

        return best_params

    def with_bayesian_search(self) -> dict:
        bs = BayesSearchCV(
            self.model,
            self.param_grid,
            cv=5,
            n_jobs=2,
            verbose=self.verbose,
            n_iter=self.n_iter,
            scoring=self.scoring
        )

        bs.fit(
            self.X_train,
            self.y_train
        )

        best_model = bs.best_estimator_
        predictions = best_model.predict(self.X_test)
        proba = bs.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        cf = classification_report(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions)
        rocauc_score = roc_auc_score(predictions, proba)
        precision = precision_score(predictions, self.y_test)
        cm = confusion_matrix(predictions, self.y_test)

        print(f"Best params: {bs.best_params_}\n")
        print("\nClassification Report\n", cf)
        print(f"Accuracy Score: {accuracy}\n")
        print(f"Recall Score: {recall}\n")
        print(f"Precision Score: {precision}\n")
        print(f"F1 Score: {f1}\n")
        print(f"\nROC AUC Score: {rocauc_score}\n")
        print(f"Confusion Matrix: {cm}")

        if (
            recall >= self.thresholds.get("recall", 0) and
            precision >= self.thresholds.get("precision", 0) and
            accuracy >= self.thresholds.get("accuracy", 0) and
            f1 >= self.thresholds.get("f1_score", 0) and
            rocauc_score >= self.thresholds.get("roc_auc", 0)
        ):
            return bs.best_params_
        else:
            print("Metrics are not good enough. Check your steps before this optimization.")
            return {}