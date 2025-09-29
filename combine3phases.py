#!/usr/bin/env python3
"""
EHDSAF Complete Pipeline v2.0
- Web Scraping
- NLP Processing
- Blockchain Storage
- Visualization
"""

import os
import json
import logging
from datetime import datetime
from web3 import Web3
from web3.exceptions import ContractLogicError
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ehdsaf.log'),
        logging.StreamHandler()
    ]
)


class ConfigManager:
    """Handles all configuration needs"""


def load_config():
    """Load config from file or environment variables"""
    try:
        # Try .env first
        load_dotenv()
        if os.getenv('INFURA_URL'):
            return {
                'infura_url': os.getenv('INFURA_URL'),
                'contract_address': os.getenv('CONTRACT_ADDRESS'),
                'contract_abi': json.loads(os.getenv('CONTRACT_ABI')),
                'wallet_address': os.getenv('WALLET_ADDRESS'),
                'wallet_private_key': os.getenv('WALLET_PRIVATE_KEY'),
                'chain_id': int(os.getenv('CHAIN_ID', 11155111)),
                'token_reward': int(os.getenv('TOKEN_REWARD', 50))
            }

        # Fallback to config file
        with open('blockchain_config.json') as f:
            return json.load(f)
    except Exception as e:
        logging.error("Configuration loading failed. Create either:")
        logging.error("1. A .env file with blockchain credentials")
        logging.error("2. A blockchain_config.json file")
        raise RuntimeError("Missing configuration") from e

class BlockchainIntegrator:
    """Phase 3: Blockchain Storage"""

    def __init__(self):
        self.config = load_config()
        self._init_web3()

    def _init_web3(self):
        """Initialize Web3 connection"""
        self.w3 = Web3(Web3.HTTPProvider(self.config['infura_url']))
        if not self.w3.is_connected():
            raise ConnectionError("Blockchain connection failed")

        self.contract = self.w3.eth.contract(
            address=self.config['contract_address'],
            abi=self.config['contract_abi']
        )

    def store_results(self, results_df, source_url):
        """Store processed results on blockchain"""
        try:
            # Prepare data
            data_hash = self._generate_hash(results_df)
            metadata = {
                'source_url': source_url,
                'detection_date': datetime.now().strftime('%Y-%m-%d'),
                'pattern_count': int(results_df['is_dark_pattern'].sum())
            }

            # Build transaction
            tx = self.contract.functions.storeDSDRecord(
                data_hash,
                metadata['source_url'],
                metadata['detection_date'],
                metadata['pattern_count']
            ).build_transaction({
                'chainId': self.config['chain_id'],
                'gas': 200000,
                'nonce': self.w3.eth.get_transaction_count(self.config['wallet_address']),
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(
                tx, self.config['wallet_private_key'])
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Blockchain storage successful: {receipt.transactionHash.hex()}")
            return receipt.transactionHash.hex()

        except ContractLogicError as e:
            logging.error(f"Smart contract error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Blockchain operation failed: {str(e)}")
            raise

    @staticmethod
    def _generate_hash(df):
        """Generate SHA-256 hash of processed data"""
        return Web3.keccak(text=df.to_json()).hex()


class DataProcessor:
    """Phase 2: NLP Processing"""

    @staticmethod
    def process_text(text):
        """Clean and preprocess text"""
        # Implement your NLP processing logic
        return {
            'original': text,
            'processed': text.lower()[:100],  # Simplified for example
            'is_dark_pattern': False,  # Replace with actual detection
            'confidence': 0.0
        }

    def analyze(self, texts):
        """Process multiple text entries"""
        results = []
        for text in texts:
            try:
                results.append(self.process_text(text))
            except Exception as e:
                logging.warning(f"Text processing failed: {str(e)}")
        return pd.DataFrame(results)


class EHDSAFPipeline:
    """Complete end-to-end pipeline"""

    def __init__(self):
        self.processor = DataProcessor()
        self.blockchain = BlockchainIntegrator()

    def run(self, source_url, sample_texts=None):
        """Execute full pipeline"""
        try:
            # Phase 1: Data Collection (simulated with sample texts)
            texts = sample_texts or [
                "Forwarded: Secret deal",
                "Normal message",
                "VIP offer via DM"
            ]

            # Phase 2: Processing
            results = self.processor.analyze(texts)
            results['timestamp'] = datetime.now()

            # Phase 3: Blockchain
            tx_hash = self.blockchain.store_results(results, source_url)
            results['blockchain_tx'] = tx_hash

            # Generate visuals
            self._generate_visuals(results)

            return results

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

    @staticmethod
    def _generate_visuals(df):
        """Create analysis visualizations"""
        try:
            os.makedirs('visualizations', exist_ok=True)

            # Detection distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x='is_dark_pattern', data=df)
            plt.title('Dark Pattern Detection')
            plt.savefig('visualizations/detection_distribution.png')
            plt.close()

            logging.info("Visualizations generated in visualizations/ folder")
        except Exception as e:
            logging.warning(f"Visualization failed: {str(e)}")


if __name__ == "__main__":
    try:
        print("""
        EHDSAF Pipeline - Configuration Guide:
        1. Create .env file with:
           INFURA_URL=https://sepolia.infura.io/v3/YOUR_KEY
           CONTRACT_ADDRESS=0x...
           WALLET_ADDRESS=0x...
           WALLET_PRIVATE_KEY=...
           CHAIN_ID=11155111
        2. OR create blockchain_config.json
        """)

        pipeline = EHDSAFPipeline()
        results = pipeline.run(
            source_url="https://example.com",
            sample_texts=[
                "Exclusive offer via DM",
                "Public announcement",
                "Forwarded confidential info"
            ]
        )

        print("\nResults:")
        print(results[['original', 'is_dark_pattern', 'blockchain_tx']])

    except Exception as e:
        logging.error(f"System error: {str(e)}")
        exit(1)