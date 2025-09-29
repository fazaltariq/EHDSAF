import re
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging


def preprocess_text(text):
    """
    Clean and normalize raw text for NLP processing
    Args:
        text (str): Raw input text
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Standardize whitespace and remove special chars
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)

    # Remove common social media artifacts
    patterns = [
        r'http\S+|www\.\S+',  # URLs
        r'@\w+',  # Mentions
        r'#\w+',  # Hashtags
        r'\bRT\b',  # Retweets
        r'\bDM\b',  # Direct messages
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


class EHDSAFProcessor:
    # ... (previous code remains the same until Phase 2)

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.tokenizer = None
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_nlp_models(self):
        """Load required NLP models with error handling"""
        try:
            # SpaCy for basic NLP

            # Transformers for classification
            model_name = "finiteautomata/bertweet-base-sentiment-analysis"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Set device (GPU if available)
            self.classifier.to(self.device)

            logging.info("NLP models loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load NLP models: {str(e)}")
            raise

    def classify_dark_patterns(self, texts, threshold=0.85):
        """
        Classify text samples as dark patterns
        Args:
            texts (list): List of raw text samples
            threshold (float): Confidence threshold
        Returns:
            pd.DataFrame: Results with classifications
        """
        if not texts:
            return pd.DataFrame()

        results = []
        for text in tqdm(texts, desc="Classifying Texts"):
            try:
                # Clean text
                clean_text = preprocess_text(text)
                if len(clean_text) < 10:  # Skip very short texts
                    continue

                # Rule-based detection first
                is_dark = self._detect_rule_based(clean_text)
                confidence = 1.0 if is_dark else 0.0

                # NLP classification if rule-based didn't catch it
                if not is_dark:
                    pred = self._classify_with_nlp(clean_text)
                    is_dark = pred['label'] == 'POSITIVE' and pred['score'] >= threshold
                    confidence = pred['score'] if is_dark else 0.0

                results.append({
                    'original_text': text,
                    'processed_text': clean_text,
                    'is_dark_pattern': is_dark,
                    'confidence': confidence,
                    'classification_method': 'rule-based' if confidence == 1.0 else 'NLP'
                })

            except Exception as e:
                logging.warning(f"Error processing text: {str(e)}")
                continue

        return pd.DataFrame(results)

    @staticmethod
    def _detect_rule_based(text):
        """Rule-based detection of common dark patterns"""
        dark_phrases = [
            'forwarded', 'via dm', 'private share', 'confidential',
            'exclusive access', 'limited time', 'secret', 'for your eyes only'
        ]
        return any(phrase in text.lower() for phrase in dark_phrases)

    def _classify_with_nlp(self, text):
        """Classify text using transformer model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred = {
            'label': self.classifier.config.id2label[torch.argmax(probs).item()],
            'score': torch.max(probs).item()
        }
        return pred

    def process_dataset(self, raw_data_path, output_path):
        """
        Complete processing pipeline for raw dataset
        Args:
            raw_data_path (str): Path to raw data file
            output_path (str): Path to save processed results
        """
        try:
            # Load raw data
            df = pd.read_csv(raw_data_path)
            texts = df['text'].tolist()

            # Process and classify
            self.load_nlp_models()
            results = self.classify_dark_patterns(texts)

            # Save results
            results.to_csv(output_path, index=False)
            logging.info(f"Saved processed data to {output_path}")

            return results

        except Exception as e:
            logging.error(f"Dataset processing failed: {str(e)}")
            raise
