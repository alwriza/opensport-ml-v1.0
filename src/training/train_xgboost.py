"""
XGBoost Model Training
Trains 5 regression models for kick quality prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Train XGBoost models for kick quality prediction"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.target_cols = ['stability', 'power', 'technique', 'balance', 'overall']
    
    def load_data(self, data_path):
        """Load training data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and targets
        feature_cols = [c for c in df.columns if c not in self.target_cols + ['video_id']]
        
        X = df[feature_cols].values
        y = df[self.target_cols].values
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(X)}")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y, feature_cols
    
    def train(self, data_path, output_dir):
        """
        Train all 5 XGBoost models
        
        Args:
            data_path (str): Path to training_data.csv
            output_dir (str): Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        X, y, feature_cols = self.load_data(data_path)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.model_config['test_size'],
            random_state=self.model_config['random_state']
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {}
        results = {}
        
        for i, target in enumerate(self.target_cols):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training XGBoost for: {target.upper()}")
            logger.info(f"{'='*60}")
            
            # Create model
            model = xgb.XGBRegressor(
                n_estimators=self.model_config['n_estimators'],
                learning_rate=self.model_config['learning_rate'],
                max_depth=self.model_config['max_depth'],
                min_child_weight=self.model_config['min_child_weight'],
                subsample=self.model_config['subsample'],
                colsample_bytree=self.model_config['colsample_bytree'],
                reg_alpha=self.model_config['reg_alpha'],
                reg_lambda=self.model_config['reg_lambda'],
                random_state=self.model_config['random_state'],
                n_jobs=-1
            )
            
            # Train with early stopping
            model.fit(
                X_train_scaled, y_train[:, i],
                eval_set=[(X_test_scaled, y_test[:, i])],
                verbose=False
            )
            
            # Predict
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Clip to valid range [0, 100]
            train_pred = np.clip(train_pred, 0, 100)
            test_pred = np.clip(test_pred, 0, 100)
            
            # Evaluate
            train_rmse = np.sqrt(mean_squared_error(y_train[:, i], train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test[:, i], test_pred))
            train_mae = mean_absolute_error(y_train[:, i], train_pred)
            test_mae = mean_absolute_error(y_test[:, i], test_pred)
            train_r2 = r2_score(y_train[:, i], train_pred)
            test_r2 = r2_score(y_test[:, i], test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train[:, i],
                cv=self.model_config['cv_folds'],
                scoring='r2',
                n_jobs=-1
            )
            
            logger.info(f"\nMetrics:")
            logger.info(f"  Train RMSE: {train_rmse:.2f}")
            logger.info(f"  Test RMSE:  {test_rmse:.2f}")
            logger.info(f"  Train MAE:  {train_mae:.2f}")
            logger.info(f"  Test MAE:   {test_mae:.2f}")
            logger.info(f"  Train R²:   {train_r2:.3f}")
            logger.info(f"  Test R²:    {test_r2:.3f}")
            logger.info(f"  CV R²:      {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")
            
            # Feature importance
            feature_importance = sorted(
                zip(feature_cols, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            logger.info(f"\nTop 10 Features:")
            for feat, imp in feature_importance:
                logger.info(f"  {feat[:50]:<50} {imp:.4f}")
            
            # Store
            models[target] = model
            results[target] = {
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_pred': test_pred,
                'test_actual': y_test[:, i],
                'feature_importance': feature_importance
            }
        
        # Save models
        joblib.dump(models, output_dir / 'xgboost_models.pkl')
        joblib.dump(scaler, output_dir / 'scaler.pkl')
        joblib.dump(feature_cols, output_dir / 'feature_names.pkl')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Models saved to {output_dir}/")
        logger.info(f"{'='*60}")
        
        # Plot results
        self.plot_results(results, output_dir)
        
        # Save metrics summary
        metrics_df = pd.DataFrame({
            'target': list(results.keys()),
            'test_r2': [r['test_r2'] for r in results.values()],
            'test_rmse': [r['test_rmse'] for r in results.values()],
            'test_mae': [r['test_mae'] for r in results.values()]
        })
        metrics_df.to_csv(output_dir / 'model_metrics.csv', index=False)
        
        return models, scaler, results
    
    def plot_results(self, results, output_dir):
        """Plot predicted vs actual for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (target, data) in enumerate(results.items()):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(data['test_actual'], data['test_pred'], alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(data['test_actual'].min(), data['test_pred'].min())
            max_val = max(data['test_actual'].max(), data['test_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Score', fontsize=12)
            ax.set_ylabel('Predicted Score', fontsize=12)
            ax.set_title(
                f'{target.capitalize()}\nR² = {data["test_r2"]:.3f}, RMSE = {data["test_rmse"]:.1f}',
                fontsize=13,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_evaluation.png', dpi=150, bbox_inches='tight')
        logger.info(f"\n✓ Evaluation plots saved to {output_dir / 'model_evaluation.png'}")
        plt.close()


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost models')
    parser.add_argument('--data', required=True, help='Path to training_data.csv')
    parser.add_argument('--output', default='models/', help='Output directory for models')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    trainer = XGBoostTrainer(config_path=args.config)
    trainer.train(args.data, args.output)
    
    logger.info("\nTraining complete! 🎉")


if __name__ == "__main__":
    main()
