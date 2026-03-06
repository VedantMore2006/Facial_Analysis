"""
XGBoost Training Pipeline

Trains mental health classification model using behavioral features.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from ml.disorder_profiles import DISORDER_LABELS


class MentalHealthClassifier:
    """
    XGBoost-based mental health classifier.
    """

    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42):
        """
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        """
        self.label_map = {
            label: name.replace('_', ' ').title()
            for name, label in DISORDER_LABELS.items()
        }

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='multi:softprob',
            num_class=len(self.label_map),
            eval_metric='mlogloss'
        )

        self.feature_names = None

    def train(self, dataset_path, test_size=0.2):
        """
        Train model on dataset.

        Parameters
        ----------
        dataset_path : str
            Path to training dataset CSV
        test_size : float
            Proportion of test set

        Returns
        -------
        dict
            Training results and metrics
        """
        print(f"\n{'='*60}")
        print("XGBOOST TRAINING")
        print(f"{'='*60}")

        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded dataset: {len(df)} samples")

        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']

        self.feature_names = list(X.columns)
        print(f"✓ Features: {len(self.feature_names)}")

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"✓ Train samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")

        # Train model
        print(f"\nTraining XGBoost classifier...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print(f"✓ Training complete")

        # Evaluate
        print(f"\nEvaluating model...")
        y_pred = self.model.predict(X_test)

        ordered_labels = sorted(self.label_map.keys())
        ordered_names = [self.label_map[label] for label in ordered_labels]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=ordered_labels).tolist(),
            'classification_report': classification_report(
                y_test,
                y_pred,
                labels=ordered_labels,
                target_names=ordered_names,
                zero_division=0
            )
        }

        # Print results
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE")
        print(f"{'='*60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")

        print(f"\n{results['classification_report']}")

        return results

    def get_feature_importance(self, top_n=20):
        """
        Get top important features.

        Parameters
        ----------
        top_n : int
            Number of top features

        Returns
        -------
        pd.DataFrame
            Feature importance ranking
        """
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\n{'='*60}")
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print(f"{'='*60}")

        for i, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:50s} {row['importance']:.4f}")

        return importance_df

    def save_model(self, model_path, metadata_path=None):
        """
        Save trained model.

        Parameters
        ----------
        model_path : str
            Path to save model (.pkl)
        metadata_path : str, optional
            Path to save metadata (.json)
        """
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved model: {model_path}")

        # Save metadata
        if metadata_path:
            metadata = {
                'feature_names': self.feature_names,
                'label_map': self.label_map,
                'num_features': len(self.feature_names),
                'num_classes': len(self.label_map),
                'model_params': self.model.get_params()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved metadata: {metadata_path}")

    def load_model(self, model_path, metadata_path=None):
        """
        Load trained model.

        Parameters
        ----------
        model_path : str
            Path to model file
        metadata_path : str, optional
            Path to metadata file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded model: {model_path}")

        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.label_map = {int(k): v for k, v in metadata['label_map'].items()}
            print(f"✓ Loaded metadata: {metadata_path}")

    def predict(self, features):
        """
        Predict mental health state from features.

        Parameters
        ----------
        features : pd.DataFrame or np.array
            Feature vector(s)

        Returns
        -------
        np.array
            Predicted labels
        """
        return self.model.predict(features)

    def predict_proba(self, features):
        """
        Get prediction probabilities.

        Parameters
        ----------
        features : pd.DataFrame or np.array
            Feature vector(s)

        Returns
        -------
        np.array
            Class probabilities
        """
        return self.model.predict_proba(features)


if __name__ == "__main__":
    """
    Example usage: Train classifier
    """
    # Create and train classifier
    classifier = MentalHealthClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05
    )

    # Train on dataset
    results = classifier.train('ml/training_dataset.csv', test_size=0.2)

    # Feature importance
    importance_df = classifier.get_feature_importance(top_n=20)

    # Save importance
    importance_df.to_csv('ml/feature_importance.csv', index=False)

    # Save model
    classifier.save_model(
        model_path='ml/mental_health_model.pkl',
        metadata_path='ml/model_metadata.json'
    )

    print(f"\n✓ Training complete")
    print(f"✓ Model saved")
    print(f"\nNext step: Run evaluate_model.py for detailed analysis")
