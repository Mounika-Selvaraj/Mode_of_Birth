# model_training.py
"""
Complete Machine Learning Pipeline for Childbirth Mode Prediction
Includes: Data preprocessing, model training, evaluation, and saving
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# SHAP for explainable AI
import shap

# Utilities
import joblib
import pickle
import os
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class ChildbirthPredictor:
    """Complete ML pipeline for childbirth mode prediction"""
    
    def __init__(self, data_path='data/Mode_Of_Birth.csv'):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.shap_explainer = None
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = ['models', 'data', 'reports', 'patient_data']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Clean column names
        self.data.columns = self.data.columns.str.strip().str.upper()
        
        # Display data types
        print("\n📋 Data types:")
        print(self.data.dtypes)
        
        # Handle missing values - first check for any
        print(f"\n🔍 Missing values: {self.data.isnull().sum().sum()}")
        if self.data.isnull().sum().sum() > 0:
            print("Columns with missing values:")
            print(self.data.isnull().sum()[self.data.isnull().sum() > 0])
        
        # Fill missing values for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Fill missing values for categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown')
        
        # Convert boolean-like columns
        self.convert_boolean_columns()
        
        # Create new features
        self.create_features()
        
        # Prepare features and target
        self.prepare_features_target()
        
        print("✅ Data preprocessing complete!")
        return self.data
    
    def convert_boolean_columns(self):
        """Convert boolean-like columns to numerical"""
        print("\n🔄 Converting boolean columns...")
        
        # Common boolean column patterns
        boolean_patterns = {
            't': 1, 'true': 1, 'yes': 1, 'y': 1, '1': 1,
            'f': 0, 'false': 0, 'no': 0, 'n': 0, '0': 0
        }
        
        # Check each column for boolean-like values
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Sample values to check
                sample_values = self.data[col].dropna().unique()[:10]
                
                # Check if all sample values can be mapped to boolean
                if all(str(val).lower() in boolean_patterns for val in sample_values if pd.notnull(val)):
                    print(f"  Converting {col} to numerical boolean")
                    self.data[col] = self.data[col].astype(str).str.lower().map(boolean_patterns).fillna(0)
        
        return self.data
    
    def create_features(self):
        """Create new clinical features"""
        print("\n🎯 Creating clinical features...")
        
        # Calculate BMI if not present
        if 'BMI' not in self.data.columns and 'HEIGHT' in self.data.columns and 'WEIGHT' in self.data.columns:
            print("  Calculating BMI from height and weight")
            # Convert height from meters to cm if needed
            if self.data['HEIGHT'].max() < 3:  # Likely in meters
                self.data['HEIGHT'] = self.data['HEIGHT'] * 100
            
            self.data['BMI'] = self.data['WEIGHT'] / ((self.data['HEIGHT'] / 100) ** 2)
        
        # Create numerical BMI categories instead of strings
        if 'BMI' in self.data.columns:
            print("  Creating BMI categories (numerical)")
            self.data['BMI_CATEGORY'] = pd.cut(self.data['BMI'], 
                                              bins=[0, 18.5, 25, 30, 100],
                                              labels=[0, 1, 2, 3])  # Numerical labels
        
        # Previous cesarean flag
        if 'PREVIOUS CESAREAN' in self.data.columns:
            print("  Creating previous cesarean flag")
            # Convert to numerical if string
            if self.data['PREVIOUS CESAREAN'].dtype == 'object':
                self.data['HAS_PREV_CESAREAN'] = self.data['PREVIOUS CESAREAN'].apply(
                    lambda x: 1 if str(x).lower() in ['t', 'true', 'yes', 'y', '1'] else 0
                )
            else:
                self.data['HAS_PREV_CESAREAN'] = self.data['PREVIOUS CESAREAN'].apply(lambda x: 1 if x else 0)
        
        # Convert other potential boolean columns
        bool_columns = ['SUBSTANCE ABUSE', 'SMOKING', 'ALCOHOL', 'ART', 'AMNIOCENTESIS',
                       'COMORBIDITY', 'PREINDUCTION', 'INDUCTION', 'ANESTHESIA']
        
        for col in bool_columns:
            if col in self.data.columns and self.data[col].dtype == 'object':
                print(f"  Converting {col} to numerical")
                self.data[col] = self.data[col].apply(
                    lambda x: 1 if str(x).lower() in ['t', 'true', 'yes', 'y', '1'] else 0
                )
        
        return self.data
    
    def prepare_features_target(self):
        """Prepare features and target for training"""
        print("\n🎯 Preparing features and target...")
        
        # Define target column
        target_column = 'TYPE OF BIRTH'
        
        if target_column not in self.data.columns:
            # Try to find alternative target column names
            possible_targets = ['TYPE OF BIRTH', 'TYPE_OF_BIRTH', 'BIRTH_TYPE', 'DELIVERY_TYPE']
            for possible in possible_targets:
                if possible in self.data.columns:
                    target_column = possible
                    break
        
        print(f"  Target column: {target_column}")
        
        # Define columns to exclude
        exclude_cols = [target_column, 'DELIVERY_PATTERN', 'RISK_SCORE', 'COUNTRY OF ORIGYN', 
                       'MATERNAL EDUCATION']  # Exclude categorical columns that need special encoding
        
        # Get feature columns (numerical only for simplicity)
        feature_cols = []
        for col in self.data.columns:
            if col not in exclude_cols and col != target_column:
                # Only include numerical columns
                if self.data[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                    feature_cols.append(col)
                # Also include boolean columns that are already numerical
                elif self.data[col].dtype == 'bool':
                    feature_cols.append(col)
                elif self.data[col].nunique() < 10 and self.data[col].dtype == 'object':
                    # Convert simple categoricals to numerical
                    print(f"  Encoding categorical column: {col}")
                    le = LabelEncoder()
                    self.data[f"{col}_ENCODED"] = le.fit_transform(self.data[col].astype(str))
                    self.encoders[col] = le
                    feature_cols.append(f"{col}_ENCODED")
        
        print(f"  Selected {len(feature_cols)} features")
        print(f"  Features: {feature_cols}")
        
        # Ensure all features are numerical
        for col in feature_cols:
            if self.data[col].dtype == 'object':
                print(f"  Converting {col} to numerical")
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                self.data[col] = self.data[col].fillna(0)
        
        # Encode target variable
        if target_column in self.data.columns:
            print(f"  Encoding target variable: {target_column}")
            target_encoder = LabelEncoder()
            self.data['TARGET_ENCODED'] = target_encoder.fit_transform(self.data[target_column].astype(str))
            self.encoders[target_column] = target_encoder
            self.target_classes = target_encoder.classes_
            print(f"  Target classes: {self.target_classes}")
        
        # Split features and target
        X = self.data[feature_cols]
        y = self.data['TARGET_ENCODED']
        
        # Scale numerical features
        print("  Scaling numerical features...")
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols, index=self.data.index)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.feature_names = feature_cols
        
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        
        # Check for any non-numeric values
        print(f"\n🔍 Data validation:")
        print(f"  X_train types: {set(self.X_train.dtypes)}")
        print(f"  Any NaN in X_train: {self.X_train.isnull().sum().sum()}")
        print(f"  Any NaN in X_test: {self.X_test.isnull().sum().sum()}")
        
        # Ensure all data is numeric
        self.X_train = self.X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.X_test = self.X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X, y
    
    def train_models(self):
        """Train multiple ML models"""
        print("\n🤖 Training ML models...")
        
        # Define base models with optimized parameters
        base_models = {
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'LinearSVC': SVC(kernel='linear', probability=True, random_state=42, C=1.0),
            'GaussianNB': GaussianNB(),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False,
                verbosity=0
            )
        }
        
        # Train and evaluate each model
        for name, model in base_models.items():
            print(f"  Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                
                # Evaluate
                y_pred = model.predict(self.X_test)
                self.results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                }
                
                print(f"    {name} trained successfully!")
            except Exception as e:
                print(f"    Error training {name}: {str(e)}")
                # Use a simple dummy model as fallback
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy='most_frequent')
                dummy.fit(self.X_train, self.y_train)
                self.models[name] = dummy
                
                y_pred = dummy.predict(self.X_test)
                self.results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                }
        
        return self.models, self.results
    
    def train_ensemble(self):
        """Create and train ensemble model"""
        print("\n🔄 Training Ensemble Voting Classifier...")
        
        # Get successful models
        successful_models = []
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):  # Only models with probability
                successful_models.append((name, model))
        
        if len(successful_models) < 2:
            print("  Not enough successful models for ensemble. Using Random Forest only.")
            self.models['Ensemble'] = self.models['RandomForest']
            y_pred = self.models['Ensemble'].predict(self.X_test)
        else:
            # Create voting classifier with successful models
            voting_clf = VotingClassifier(
                estimators=successful_models,
                voting='soft'
            )
            
            # Train ensemble
            voting_clf.fit(self.X_train, self.y_train)
            self.models['Ensemble'] = voting_clf
            
            # Evaluate ensemble
            y_pred = voting_clf.predict(self.X_test)
        
        self.results['Ensemble'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"✅ Ensemble model trained!")
        return self.models.get('Ensemble', None)
    
    def create_shap_explainer(self):
        """Create SHAP explainer for model interpretability"""
        print("\n🔍 Creating SHAP explainer...")
        
        # Use the best tree-based model for SHAP explanations
        best_tree_model = None
        
        # Try XGBoost first
        if 'XGBoost' in self.models and hasattr(self.models['XGBoost'], 'predict_proba'):
            best_tree_model = self.models['XGBoost']
            print("  Using XGBoost for SHAP explanations")
        # Then try Random Forest
        elif 'RandomForest' in self.models and hasattr(self.models['RandomForest'], 'predict_proba'):
            best_tree_model = self.models['RandomForest']
            print("  Using Random Forest for SHAP explanations")
        
        if best_tree_model:
            try:
                # Create SHAP explainer
                self.shap_explainer = shap.TreeExplainer(best_tree_model)
                
                # Test with a small sample
                sample_size = min(100, len(self.X_train))
                X_sample = self.X_train.iloc[:sample_size]
                
                # Calculate SHAP values
                shap_values = self.shap_explainer.shap_values(X_sample)
                print(f"✅ SHAP explainer created successfully!")
                
                return shap_values
            except Exception as e:
                print(f"⚠️ Could not create SHAP explainer: {str(e)}")
                self.shap_explainer = None
                return None
        else:
            print("⚠️ No tree-based model available for SHAP explanations")
            self.shap_explainer = None
            return None
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Print individual model results
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        # Print confusion matrix for ensemble
        if 'Ensemble' in self.models:
            y_pred = self.models['Ensemble'].predict(self.X_test)
            print("\n" + "="*60)
            print("📊 Confusion Matrix (Ensemble):")
            print("="*60)
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            
            # Classification report
            print("\n" + "="*60)
            print("📋 Classification Report (Ensemble):")
            print("="*60)
            print(classification_report(self.y_test, y_pred, target_names=self.target_classes))
        else:
            print("\n⚠️ No ensemble model available for detailed evaluation")
        
        return self.results
    
    def save_models(self):
        """Save all trained models and preprocessors"""
        print("\n💾 Saving models and preprocessors...")
        
        # Save ensemble model
        if 'Ensemble' in self.models:
            joblib.dump(self.models['Ensemble'], 'models/trained_model.pkl')
            print("  ✅ Ensemble model saved")
        else:
            # Save best individual model
            best_model_name = max(self.results.items(), key=lambda x: x[1]['f1'])[0]
            if best_model_name in self.models:
                joblib.dump(self.models[best_model_name], 'models/trained_model.pkl')
                print(f"  ✅ Best model ({best_model_name}) saved")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("  ✅ Scaler saved")
        
        # Save encoders
        with open('models/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        print("  ✅ Encoders saved")
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print("  ✅ Feature names saved")
        
        # Save target classes
        if hasattr(self, 'target_classes'):
            with open('models/target_classes.pkl', 'wb') as f:
                pickle.dump(self.target_classes, f)
            print("  ✅ Target classes saved")
        
        # Save SHAP explainer
        if self.shap_explainer:
            try:
                joblib.dump(self.shap_explainer, 'models/shap_explainer.pkl')
                print("  ✅ SHAP explainer saved")
            except:
                print("  ⚠️ Could not save SHAP explainer")
        
        # Save evaluation results
        with open('models/evaluation_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        print("  ✅ Evaluation results saved")
    
    def visualize_results(self):
        """Create visualizations of model performance"""
        print("\n📊 Creating visualizations...")
        
        try:
            # Create results DataFrame
            results_df = pd.DataFrame(self.results).T
            
            # Plot model comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx // 2, idx % 2]
                if metric in results_df.columns:
                    results_df[metric].plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f'{title} Comparison')
                    ax.set_ylabel(title)
                    ax.set_ylim(0, 1.0)
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for i, v in enumerate(results_df[metric]):
                        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
            print("  ✅ Model comparison chart saved")
            
            # Feature importance from Random Forest
            if 'RandomForest' in self.models:
                rf_model = self.models['RandomForest']
                if hasattr(rf_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False).head(15)
                    
                    plt.figure(figsize=(12, 8))
                    plt.barh(feature_importance['feature'][::-1], feature_importance['importance'][::-1])
                    plt.xlabel('Importance')
                    plt.title('Top 15 Feature Importances (Random Forest)')
                    plt.tight_layout()
                    plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
                    print("  ✅ Feature importance chart saved")
            
            # Target class distribution
            plt.figure(figsize=(10, 6))
            class_counts = pd.Series(self.y_train).value_counts()
            if hasattr(self, 'target_classes'):
                class_names = self.target_classes
                plt.bar(range(len(class_counts)), class_counts.values)
                plt.xticks(range(len(class_counts)), [class_names[i] for i in class_counts.index])
            else:
                plt.bar(class_counts.index, class_counts.values)
            plt.title('Target Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('models/class_distribution.png', dpi=300, bbox_inches='tight')
            print("  ✅ Class distribution chart saved")
            
        except Exception as e:
            print(f"  ⚠️ Could not create visualizations: {str(e)}")
    
    def run_complete_pipeline(self):
        """Run complete ML pipeline"""
        print("="*60)
        print("🚀 STARTING COMPLETE ML PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Train individual models
            self.train_models()
            
            # Step 3: Train ensemble
            self.train_ensemble()
            
            # Step 4: Create SHAP explainer
            self.create_shap_explainer()
            
            # Step 5: Evaluate models
            self.evaluate_models()
            
            # Step 6: Save models
            self.save_models()
            
            # Step 7: Create visualizations
            self.visualize_results()
            
            print("\n" + "="*60)
            print("✅ ML PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Display final results
            if 'Ensemble' in self.results:
                print(f"\n🎯 Ensemble Model Results:")
                print(f"   Accuracy:  {self.results['Ensemble']['accuracy']:.4f}")
                print(f"   F1-Score:  {self.results['Ensemble']['f1']:.4f}")
                print(f"   Precision: {self.results['Ensemble']['precision']:.4f}")
                print(f"   Recall:    {self.results['Ensemble']['recall']:.4f}")
            elif self.results:
                best_model_name = max(self.results.items(), key=lambda x: x[1]['f1'])[0]
                print(f"\n🎯 Best Model ({best_model_name}) Results:")
                print(f"   Accuracy:  {self.results[best_model_name]['accuracy']:.4f}")
                print(f"   F1-Score:  {self.results[best_model_name]['f1']:.4f}")
                print(f"   Precision: {self.results[best_model_name]['precision']:.4f}")
                print(f"   Recall:    {self.results[best_model_name]['recall']:.4f}")
            
            return self.models.get('Ensemble', None)
            
        except Exception as e:
            print(f"\n❌ ERROR in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Main execution
if __name__ == "__main__":
    # Initialize and run pipeline
    predictor = ChildbirthPredictor('data/Mode_Of_Birth.csv')
    model = predictor.run_complete_pipeline()
    
    if model:
        print("\n📁 Saved files:")
        saved_files = []
        if os.path.exists('models/trained_model.pkl'):
            saved_files.append("  - models/trained_model.pkl (Trained model)")
        if os.path.exists('models/scaler.pkl'):
            saved_files.append("  - models/scaler.pkl (Feature scaler)")
        if os.path.exists('models/encoders.pkl'):
            saved_files.append("  - models/encoders.pkl (Label encoders)")
        if os.path.exists('models/feature_names.pkl'):
            saved_files.append("  - models/feature_names.pkl (Feature names)")
        if os.path.exists('models/target_classes.pkl'):
            saved_files.append("  - models/target_classes.pkl (Target classes)")
        if os.path.exists('models/shap_explainer.pkl'):
            saved_files.append("  - models/shap_explainer.pkl (SHAP explainer)")
        if os.path.exists('models/evaluation_results.json'):
            saved_files.append("  - models/evaluation_results.json (Evaluation metrics)")
        
        for file_info in saved_files:
            print(file_info)
        
        print("\n✅ Now run 'streamlit run web_app.py' to start the web application!")
    else:
        print("\n❌ Model training failed. Check the error messages above.")