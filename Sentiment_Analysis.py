import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vader_sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class MockDataProcessor:
    """Mock class to simulate Phase 2 functionality"""

    def scrape_dark_content(self, url: str) -> List[str]:
        return [
            "Confidential offer - 50% discount for selected members",
            "Warning: This private deal might be a scam",
            "Normal communication with no special content",
            "Urgent: Forward this secret opportunity to friends"
        ]

    def analyze(self, texts: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            'original_text': texts,
            'processed_text': [t.lower() for t in texts],
            'is_dark_pattern': [True, True, False, True],
            'confidence': [0.95, 0.87, 0.12, 0.91]
        })


class MockBlockchainIntegrator:
    """Mock class to simulate Phase 3 functionality"""

    def store_results(self, df: pd.DataFrame, source_url: str) -> str:
        logging.info(f"Mock blockchain storage for {len(df)} records")
        return "0xmockedblockchaintxhash"


class MockModelSelector:
    """Mock class to simulate Phase 4 functionality"""

    def run(self, df: pd.DataFrame) -> object:
        class MockModel:
            def predict(self, texts):
                return [1 if "confidential" in t.lower() or "secret" in t.lower() else 0 for t in texts]

        return MockModel()


class SentimentAnalyzer:
    """
    Phase 5: Real-time Sentiment Analysis for Dark Social Data
    Features:
    - TextBlob sentiment analysis
    - VADER sentiment analysis (optimized for social media)
    - Sentiment trend tracking
    - Visualization
    """

    def __init__(self, config: Dict = None):
        self.config = config or {
            'sentiment_thresholds': {
                'positive': 0.3,
                'negative': -0.3
            },
            'output_dir': 'sentiment_results',
            'update_interval': 3600  # 1 hour in seconds
        }

        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_history = defaultdict(list)

        logging.info("SentimentAnalyzer initialized successfully")

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Perform multi-method sentiment analysis
        Returns:
            {
                'text': str,
                'timestamp': str,
                'textblob': {'polarity': float, 'subjectivity': float},
                'vader': {'pos': float, 'neu': float, 'neg': float, 'compound': float},
                'final_sentiment': str ('positive', 'neutral', 'negative')
            }
        """
        analysis = {
            'text': text,
            'timestamp': datetime.now().isoformat()
        }

        # TextBlob Analysis
        blob = TextBlob(text)
        analysis['textblob'] = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

        # VADER Analysis
        vader_scores = self.vader.polarity_scores(text)
        analysis['vader'] = vader_scores

        # Combined decision
        compound_score = vader_scores['compound']
        if compound_score >= self.config['sentiment_thresholds']['positive']:
            analysis['final_sentiment'] = 'positive'
        elif compound_score <= self.config['sentiment_thresholds']['negative']:
            analysis['final_sentiment'] = 'negative'
        else:
            analysis['final_sentiment'] = 'neutral'

        return analysis

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]

    def update_sentiment_history(self, analyses: List[Dict]):
        """Maintain historical sentiment trends"""
        for analysis in analyses:
            self.sentiment_history[analysis['final_sentiment']].append({
                'text': analysis['text'],
                'timestamp': analysis['timestamp'],
                'score': analysis['vader']['compound']
            })

    def generate_sentiment_report(self) -> str:
        """Create visual sentiment report"""
        try:
            # Prepare data for visualization
            sentiment_data = []
            for sentiment_type in ['positive', 'neutral', 'negative']:
                for entry in self.sentiment_history[sentiment_type]:
                    sentiment_data.append({
                        'sentiment': sentiment_type,
                        'score': entry['score'],
                        'timestamp': pd.to_datetime(entry['timestamp'])
                    })

            df = pd.DataFrame(sentiment_data)

            # Time series plot
            plt.figure(figsize=(12, 6))
            for sentiment_type in ['positive', 'neutral', 'negative']:
                subset = df[df['sentiment'] == sentiment_type]
                if not subset.empty:
                    plt.plot(subset['timestamp'], subset['score'], 'o',
                             label=sentiment_type, alpha=0.7)

            plt.title('Sentiment Trend Over Time')
            plt.ylabel('Sentiment Score (VADER Compound)')
            plt.xlabel('Time')
            plt.legend()
            plt.grid(True)

            report_path = os.path.join(self.config['output_dir'], 'sentiment_trend.png')
            plt.savefig(report_path)
            plt.close()

            # Save raw data
            data_path = os.path.join(self.config['output_dir'], 'sentiment_data.csv')
            df.to_csv(data_path, index=False)

            return report_path

        except Exception as e:
            logging.error(f"Report generation failed: {str(e)}")
            raise

    def run(self, texts: List[str]) -> Dict:
        """
        Execute complete sentiment analysis pipeline
        Args:
            texts: List of texts to analyze
        Returns:
            {
                'analyses': List[Dict],  # Raw sentiment results
                'report_path': str,      # Path to generated visualization
                'sentiment_distribution': Dict  # Count by sentiment type
            }
        """
        try:
            # Analyze provided texts
            analyses = self.analyze_batch(texts)

            # Update history and generate report
            self.update_sentiment_history(analyses)
            report_path = self.generate_sentiment_report()

            # Calculate distribution
            dist = {
                'positive': len(self.sentiment_history['positive']),
                'neutral': len(self.sentiment_history['neutral']),
                'negative': len(self.sentiment_history['negative'])
            }

            return {
                'analyses': analyses,
                'report_path': report_path,
                'sentiment_distribution': dist
            }

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            raise


class EHDSAF_Pipeline:
    """Complete pipeline with all phases including sentiment analysis"""

    def __init__(self):
        self.preprocessor = MockDataProcessor()  # Replace with your actual Phase 2
        self.blockchain = MockBlockchainIntegrator()  # Replace with your actual Phase 3
        self.model_selector = MockModelSelector()  # Replace with your actual Phase 4
        self.sentiment_analyzer = SentimentAnalyzer()

    def run_full_analysis(self, source_url: str) -> Dict:
        """End-to-end execution with all phases"""
        try:
            logging.info(f"Starting analysis for {source_url}")

            # Phase 1-2: Data Collection and Processing
            raw_data = self.preprocessor.scrape_dark_content(source_url)
            results_df = self.preprocessor.analyze(raw_data)

            # Phase 3: Blockchain Storage
            tx_hash = self.blockchain.store_results(results_df, source_url)
            results_df['blockchain_tx'] = tx_hash

            # Phase 4: Model Selection
            best_model = self.model_selector.run(results_df)

            # Phase 5: Sentiment Analysis
            sentiment_results = self.sentiment_analyzer.run(
                texts=results_df['original_text'].tolist()
            )

            # Combine all results
            return {
                'detection_results': results_df.to_dict('records'),
                'sentiment_analysis': sentiment_results,
                'model_metadata': {
                    'model_type': 'MockModel',  # Replace with actual model type
                    'save_path': 'models/mock_model.pkl'
                }
            }

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    try:
        print("=== EHDSAF Complete Pipeline Demo ===")

        # Initialize pipeline
        pipeline = EHDSAF_Pipeline()

        # Run analysis (using mock URL)
        results = pipeline.run_full_analysis("https://example-dark-site.com")

        # Display results
        print("\n=== Detection Results ===")
        for i, item in enumerate(results['detection_results'][:3]):  # Show first 3
            print(f"\n{i + 1}. {item['original_text'][:50]}...")
            print(f"Classification: {'Dark Pattern' if item['is_dark_pattern'] else 'Normal'}")
            print(f"Confidence: {item['confidence']:.2f}")

        print("\n=== Sentiment Analysis ===")
        sentiment = results['sentiment_analysis']
        print(f"\nPositive: {sentiment['sentiment_distribution']['positive']}")
        print(f"Neutral: {sentiment['sentiment_distribution']['neutral']}")
        print(f"Negative: {sentiment['sentiment_distribution']['negative']}")
        print(f"\nReport generated at: {sentiment['report_path']}")

        # Show sample sentiment analysis
        print("\n=== Sample Sentiment Results ===")
        for i, analysis in enumerate(sentiment['analyses'][:2]):  # Show first 2
            print(f"\n{i + 1}. {analysis['text'][:50]}...")
            print(f"Sentiment: {analysis['final_sentiment'].upper()}")
            print(f"VADER Compound: {analysis['vader']['compound']:.2f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    # Create required directories
    os.makedirs('sentiment_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Run the program
    exit_code = main()
    exit(exit_code)