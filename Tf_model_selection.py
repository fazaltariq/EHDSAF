import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Tuple, List
from collections import Counter


class TensorFlowModelSelector:
    """
    Phase 4: Automated Model Selection using TensorFlow/Keras
    Implements neural network alternatives to traditional ML models
    """

    def __init__(self, config: Dict = None):
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'model_architectures': {
                'shallow_net': {
                    'layers': [
                        {'units': 64, 'activation': 'relu', 'input_dim': 100},  # input_dim should match feature size
                        {'units': 1, 'activation': 'sigmoid'}
                    ],
                    'optimizer': {'learning_rate': 0.001}
                },
                'deep_net': {
                    'layers': [
                        {'units': 128, 'activation': 'relu', 'input_dim': 100},
                        {'units': 64, 'activation': 'relu'},
                        {'units': 32, 'activation': 'relu'},
                        {'units': 1, 'activation': 'sigmoid'}
                    ],
                    'optimizer': {'learning_rate': 0.0005}
                }
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'patience': 5
            },
            'model_save_path': 'models/best_model.keras'
        }

        os.makedirs('models', exist_ok=True)
        self.best_model = None
        self.vocab = None
        self.max_len = 0

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _build_vocabulary(self, texts: List[str]) -> None:
        """Create vocabulary from all texts"""
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        self.vocab = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 reserved for padding
        self.max_len = max(len(text.split()) for text in texts)

    def _text_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded numerical sequences"""
        if not self.vocab:
            self._build_vocabulary(texts)

        sequences = []
        for text in texts:
            seq = [self.vocab.get(word, 0) for word in text.split()]  # 0 for OOV
            sequences.append(seq)

        # Padding
        padded = np.zeros((len(sequences), self.max_len))
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq[:self.max_len]

        return padded

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to features and labels"""
        try:
            texts = df['processed_text'].tolist()
            X = self._text_to_sequences(texts)
            y = df['is_dark_pattern'].values.astype(np.float32)

            logging.info(f"Data prepared - Features: {X.shape}, Labels: {y.shape}")
            return X, y

        except Exception as e:
            logging.error(f"Data preparation failed: {str(e)}")
            raise

    def _build_model(self, architecture: Dict, input_shape: int) -> tf.keras.Model:
        """Construct a Keras model from config"""
        model = Sequential()

        # First layer needs input shape
        first_layer = architecture['layers'][0].copy()
        first_layer['input_dim'] = input_shape
        model.add(Dense(**first_layer))

        # Add remaining layers
        for layer in architecture['layers'][1:]:
            model.add(Dense(**layer))

        model.compile(
            optimizer=Adam(**architecture['optimizer']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and evaluate all configured models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

        results = {}
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['patience'],
            restore_best_weights=True
        )

        input_shape = X_train.shape[1]

        for model_name, architecture in self.config['model_architectures'].items():
            try:
                logging.info(f"\nTraining {model_name}...")

                model = self._build_model(architecture, input_shape)

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=self.config['training']['epochs'],
                    batch_size=self.config['training']['batch_size'],
                    callbacks=[early_stop],
                    verbose=1  # Show progress
                )

                # Evaluate
                test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
                results[model_name] = {
                    'model': model,
                    'accuracy': test_acc,
                    'auc': test_auc,
                    'history': history.history
                }

                logging.info(f"{model_name} results - Accuracy: {test_acc:.3f}, AUC: {test_auc:.3f}")

            except Exception as e:
                logging.warning(f"Failed to train {model_name}: {str(e)}")
                continue

        return results

    def select_best_model(self, results: Dict) -> tf.keras.Model:
        """Select model with highest AUC score"""
        if not results:
            raise ValueError("No models were successfully trained")

        best_model_name = max(
            results.keys(),
            key=lambda x: results[x]['auc']
        )

        self.best_model = results[best_model_name]['model']
        logging.info(f"Selected {best_model_name} as best model (AUC: {results[best_model_name]['auc']:.3f})")

        return self.best_model

    def save_model(self, path: str = None) -> str:
        """Save best model to disk"""
        if not self.best_model:
            raise RuntimeError("No model has been trained")

        save_path = path or self.config['model_save_path']
        self.best_model.save(save_path)
        logging.info(f"Model saved to {os.path.abspath(save_path)}")
        return save_path

    def run(self, df: pd.DataFrame) -> tf.keras.Model:
        """Complete model selection pipeline"""
        try:
            # 1. Prepare data
            X, y = self.prepare_data(df)

            # 2. Train models
            results = self.train_models(X, y)

            # 3. Select best
            self.select_best_model(results)

            # 4. Persist model
            self.save_model()

            return self.best_model

        except Exception as e:
            logging.error(f"Model selection failed: {str(e)}")
            raise


# Test Execution
if __name__ == "__main__":
    # Configure logging to show in console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Sample test data
    test_data = pd.DataFrame({
        'processed_text': [
            "forwarded secret deal",
            "normal message",
            "exclusive vip offer",
            "public announcement",
            "private limited offer",
            "official communication",
            "confidential information",
            "general update"
        ],
        'is_dark_pattern': [1, 0, 1, 0, 1, 0, 1, 0]
    })

    # Initialize and run
    try:
        print("=== Starting TensorFlow Model Selection ===")
        selector = TensorFlowModelSelector()
        best_model = selector.run(test_data)

        # Test prediction
        sample_texts = ["private limited offer", "normal message"]
        sample_features = selector._text_to_sequences(sample_texts)
        predictions = best_model.predict(sample_features)

        print("\n=== Prediction Results ===")
        for text, pred in zip(sample_texts, predictions):
            confidence = pred[0]
            classification = "Dark Pattern" if confidence > 0.5 else "Normal"
            print(f"Text: '{text[:30]}...'")
            print(f"Classification: {classification} (Confidence: {confidence:.2f})")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {str(e)}")