"""
Model Evaluation & Visualization

Comprehensive evaluation of trained mental health classifier.
Generates visualizations and detailed performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
import json


class ModelEvaluator:
    """
    Evaluates and visualizes classifier performance.
    """

    def __init__(self, model_path, metadata_path, dataset_path):
        """
        Parameters
        ----------
        model_path : str
            Path to trained model
        metadata_path : str
            Path to model metadata
        dataset_path : str
            Path to test dataset
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_map = {int(k): v for k, v in metadata['label_map'].items()}

        # Load dataset
        df = pd.read_csv(dataset_path)
        X = df.drop('label', axis=1)
        y = df['label']

        # Split (use same random state as training)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✓ Loaded model and dataset")
        print(f"  Test samples: {len(self.X_test)}")

    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix heatmap.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        y_pred = self.model.predict(self.X_test)
        ordered_labels = sorted(self.label_map.keys())
        ordered_names = [self.label_map[label] for label in ordered_labels]
        cm = confusion_matrix(self.y_test, y_pred, labels=ordered_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=ordered_names,
                   yticklabels=ordered_names)
        
        plt.title('Confusion Matrix - Mental Health Classification', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {save_path}")

        plt.show()

    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot top feature importance.

        Parameters
        ----------
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save plot
        """
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature importance: {save_path}")

        plt.show()

    def plot_class_performance(self, save_path=None):
        """
        Plot per-class performance metrics.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        y_pred = self.model.predict(self.X_test)

        ordered_labels = sorted(self.label_map.keys())
        ordered_names = [self.label_map[label] for label in ordered_labels]

        # Compute per-class metrics
        precision = precision_score(self.y_test, y_pred, labels=ordered_labels, average=None, zero_division=0)
        recall = recall_score(self.y_test, y_pred, labels=ordered_labels, average=None, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, labels=ordered_labels, average=None, zero_division=0)

        # Create DataFrame
        metrics_df = pd.DataFrame({
            'Class': ordered_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.25

        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='skyblue')
        ax.bar(x, metrics_df['Recall'], width, label='Recall', color='lightcoral')
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='lightgreen')

        ax.set_xlabel('Mental Health State', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Class'])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved class performance: {save_path}")

        plt.show()

    def plot_prediction_confidence(self, save_path=None):
        """
        Plot prediction confidence distribution.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        y_pred_proba = self.model.predict_proba(self.X_test)
        max_proba = np.max(y_pred_proba, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=14, pad=20)
        plt.axvline(0.8, color='red', linestyle='--', label='High Confidence (0.8)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confidence distribution: {save_path}")

        plt.show()

        # Print confidence stats
        print(f"\n{'='*60}")
        print("PREDICTION CONFIDENCE STATISTICS")
        print(f"{'='*60}")
        print(f"Mean confidence: {max_proba.mean():.4f}")
        print(f"Median confidence: {np.median(max_proba):.4f}")
        print(f"High confidence (>0.8): {(max_proba > 0.8).sum()} / {len(max_proba)} ({(max_proba > 0.8).mean()*100:.1f}%)")
        print(f"Low confidence (<0.5): {(max_proba < 0.5).sum()} / {len(max_proba)} ({(max_proba < 0.5).mean()*100:.1f}%)")

    def generate_full_report(self, output_dir='ml/evaluation'):
        """
        Generate complete evaluation report with all visualizations.

        Parameters
        ----------
        output_dir : str
            Directory to save reports
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("GENERATING FULL EVALUATION REPORT")
        print(f"{'='*60}")

        # Plot all visualizations
        self.plot_confusion_matrix(save_path=f'{output_dir}/confusion_matrix.png')
        self.plot_feature_importance(top_n=20, save_path=f'{output_dir}/feature_importance.png')
        self.plot_class_performance(save_path=f'{output_dir}/class_performance.png')
        self.plot_prediction_confidence(save_path=f'{output_dir}/confidence_distribution.png')

        # Generate text report
        y_pred = self.model.predict(self.X_test)
        ordered_labels = sorted(self.label_map.keys())
        ordered_names = [self.label_map[label] for label in ordered_labels]
        report = classification_report(
            self.y_test,
            y_pred,
            labels=ordered_labels,
            target_names=ordered_names,
            digits=4,
            zero_division=0
        )

        report_path = f'{output_dir}/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MENTAL HEALTH CLASSIFICATION - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
            f.write("\n\n")
            f.write("="*60 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("="*60 + "\n")
            cm = confusion_matrix(self.y_test, y_pred, labels=ordered_labels)
            f.write(str(cm))

        print(f"\n✓ Full evaluation report generated: {output_dir}")
        print(f"  - Confusion matrix plot")
        print(f"  - Feature importance plot")
        print(f"  - Class performance plot")
        print(f"  - Confidence distribution plot")
        print(f"  - Text report")


if __name__ == "__main__":
    """
    Example usage: Evaluate trained model
    """
    evaluator = ModelEvaluator(
        model_path='ml/mental_health_model.pkl',
        metadata_path='ml/model_metadata.json',
        dataset_path='ml/training_dataset.csv'
    )

    # Generate full report
    evaluator.generate_full_report(output_dir='ml/evaluation')

    print(f"\n✓ Evaluation complete")
