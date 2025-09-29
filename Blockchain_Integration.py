import json
import hashlib
from web3 import Web3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging


class BlockchainIntegrator:
    """
    Phase 3: Secure Dark Social Data (DSD) sharing via blockchain
    Features:
    - Ethereum smart contract integration
    - IPFS storage for large datasets
    - Token incentives for data sharing
    - Tamper-proof audit trails
    """

    def __init__(self, config_file='blockchain_config.json'):
        try:
            # Load blockchain configuration
            with open(config_file) as f:
                self.config = json.load(f)

            # Initialize Web3 connection
            self.w3 = Web3(Web3.HTTPProvider(self.config['infura_url']))
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to Ethereum network")

            # Load contract ABI and address
            with open(self.config['contract_abi_path']) as f:
                self.contract_abi = json.load(f)

            self.contract = self.w3.eth.contract(
                address=self.config['contract_address'],
                abi=self.contract_abi
            )

            # Set default account
            self.account = self.config['wallet_address']
            self.private_key = self.config['wallet_private_key']

            logging.info("Blockchain module initialized successfully")

        except Exception as e:
            logging.error(f"Blockchain initialization failed: {str(e)}")
            raise

    def _sign_transaction(self, tx):
        """Helper method to sign transactions"""
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        return signed_tx

    def store_on_blockchain(self, data_hash, metadata):
        """
        Store data reference on Ethereum blockchain
        Args:
            data_hash (str): IPFS CID or data fingerprint
            metadata (dict): {
                'source_url': str,
                'detection_date': str,
                'dark_pattern_count': int
            }
        Returns:
            str: Transaction hash
        """
        try:
            # Build transaction
            tx = self.contract.functions.storeDSDRecord(
                data_hash,
                metadata['source_url'],
                metadata['detection_date'],
                metadata['dark_pattern_count']
            ).build_transaction({
                'chainId': self.config['chain_id'],
                'gas': 200000,
                'gasPrice': self.w3.to_wei('50', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account),
            })

            # Sign and send
            signed_tx = self._sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Transaction mined: {receipt.transactionHash.hex()}")

            # Issue tokens as incentive
            self._issue_tokens(receipt)

            return receipt.transactionHash.hex()

        except Exception as e:
            logging.error(f"Blockchain storage failed: {str(e)}")
            raise

    def _issue_tokens(self, receipt):
        """Issue incentive tokens to data contributor"""
        try:
            tx = self.contract.functions.issueTokens(
                self.account,
                self.config['token_reward_amount']
            ).build_transaction({
                'chainId': self.config['chain_id'],
                'gas': 100000,
                'gasPrice': self.w3.to_wei('50', 'gwei'),
                'nonce': self.w3.eth.get_transaction_count(self.account),
            })

            signed_tx = self._sign_transaction(tx)
            self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        except Exception as e:
            logging.warning(f"Token issuance failed: {str(e)}")

    def verify_data(self, data_hash):
        """Verify data integrity on blockchain"""
        record = self.contract.functions.getDSDRecord(data_hash).call()
        return {
            'exists': record[0],
            'source_url': record[1],
            'timestamp': record[2],
            'pattern_count': record[3]
        }

    def generate_visualizations(self, results_df, output_dir='visualizations'):
        """Generate analytics dashboards"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            # 1. Detection Distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x='is_dark_pattern', data=results_df, palette=['#4CAF50', '#F44336'])
            plt.title('Dark Pattern Detection Distribution')
            plt.savefig(f'{output_dir}/detection_distribution.png')
            plt.close()

            # 2. Confidence Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=results_df[results_df['is_dark_pattern']],
                         x='confidence', bins=10, color='#FF5722')
            plt.title('Confidence Score Distribution for Dark Patterns')
            plt.savefig(f'{output_dir}/confidence_distribution.png')
            plt.close()

            # 3. Timeline Analysis
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            plt.figure(figsize=(12, 6))
            results_df.set_index('timestamp')['is_dark_pattern'].resample('D').sum().plot()
            plt.title('Daily Dark Pattern Detections')
            plt.savefig(f'{output_dir}/daily_detections.png')
            plt.close()

            logging.info(f"Visualizations saved to {output_dir}")

        except Exception as e:
            logging.error(f"Visualization generation failed: {str(e)}")
            raise


class Preprocessing_NLP:
    def scrape_dark_content(self, source_url):
        pass


class EHDSAF_Pipeline:
    """Complete pipeline integrating all phases"""

    def __init__(self):
        self.preprocessor = Preprocessing_NLP()
        self.blockchain = BlockchainIntegrator()

    def run_full_analysis(self, source_url):
        """End-to-end execution"""
        try:
            # Phase 1: Data Collection
            raw_data = self.preprocessor.scrape_dark_content(source_url)

            # Phase 2: Analysis
            results_df = self.preprocessor.detect_dark_patterns(raw_data)
            results_df['timestamp'] = datetime.now().isoformat()

            # Phase 3: Blockchain Storage
            data_hash = hashlib.sha256(
                results_df.to_json().encode()
            ).hexdigest()

            metadata = {
                'source_url': source_url,
                'detection_date': datetime.now().strftime('%Y-%m-%d'),
                'dark_pattern_count': int(results_df['is_dark_pattern'].sum())
            }

            tx_hash = self.blockchain.store_on_blockchain(data_hash, metadata)
            results_df['blockchain_tx'] = tx_hash

            # Generate visualizations
            self.blockchain.generate_visualizations(results_df)

            return results_df

        except Exception as e:
            logging.error(f"Pipeline execution failed: {str(e)}")
            raise


# Example Usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ehdsaf_pipeline.log'),
            logging.StreamHandler()
        ]
    )

    try:
        pipeline = EHDSAF_Pipeline()
        results = pipeline.run_full_analysis("https://example.com")

        print("\n=== Analysis Results ===")
        print(results[['original_text', 'is_dark_pattern', 'blockchain_tx']].head())
        print("\nCheck visualizations/ folder for charts")

    except Exception as e:
        logging.error(f"System error: {str(e)}")