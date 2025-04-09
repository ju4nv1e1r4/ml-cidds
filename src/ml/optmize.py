from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


class Optimize:
    def __init__(self, model, param_grid, X_train, X_test, y_train, y_test, cv, scoring="recall"):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.scoring = scoring

    def with_random_search(self):
        rs = RandomizedSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            n_jobs=2,
            verbose=1,
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

    def with_bayesian_search(self):
        bs = BayesSearchCV(
            self.model,
            self.param_grid,
            cv=5,
            n_jobs=2,
            verbose=1,
            n_iter=50,
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
        rocauc_score = roc_auc_score(predictions, proba)
        precision = precision_score(predictions, self.y_test)
        cm = confusion_matrix(predictions, self.y_test)

        print(f"Best params: {bs.best_params_}\n")
        print("\nClassification Report\n", cf)
        print(f"Accuracy Score: {accuracy}\n")
        print(f"Recall Score: {recall}\n")
        print(f"Precision Score: {precision}\n")
        print(f"\nROC AUC Score: {rocauc_score}\n")
        print(f"Confusion Matrix: {cm}")