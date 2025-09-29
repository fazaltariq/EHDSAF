#!/usr/bin/env python3
"""
EHDSAF Pipeline - Dark Social Content Analyzer
Fixed and fully executable version
"""

import os
import re
import logging
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import torch

# Environment configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_PROGRESS_BAR'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class EHDSAFProcessor:
    def __init__(self, headless=True):
        """Initialize with Chrome options and NLP models"""
        self.driver = None
        self.device = None
        self.tokenizer = None
        self.dark_detector = None

        try:
            self._init_webdriver(headless)
            self._init_nlp_models()
            logging.info("EHDSAF Processor initialized successfully")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            self._cleanup_resources()
            raise

    def _init_webdriver(self, headless):
        """Configure Chrome with anti-detection settings"""
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])

        if headless:
            options.add_argument('--headless=new')

        service = ChromeService(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)

    def _init_nlp_models(self):
        """Load NLP models with GPU acceleration if available"""
        self.device = 0 if torch.cuda.is_available() else -1

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.dark_detector = pipeline(
                "text-classification",
                model="finiteautomata/bertweet-base-sentiment-analysis",
                device=self.device,
                framework="pt"
            )
        except Exception as e:
            logging.error(f"Failed to load NLP models: {str(e)}")
            raise

    def _cleanup_resources(self):
        """Release all allocated resources"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def scrape_dark_content(self, url, scroll_attempts=3, wait_time=2):
        """Scrape potentially dark social content"""
        if not self.driver:
            raise RuntimeError("WebDriver not initialized")

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body')))

            last_height = self.driver.execute_script("return document.body.scrollHeight")
            for _ in range(scroll_attempts):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(wait_time)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return [elem.get_text(strip=True) for elem in soup.find_all(['div', 'span', 'p'])]

        except Exception as e:
            logging.error(f"Scraping failed for {url}: {str(e)}")
            return []

    @staticmethod
    def _clean_text(text):
        """Advanced text cleaning pipeline"""
        if not isinstance(text, str):
            return ""

        patterns = [
            (r'http\S+|www\.\S+', ''),
            (r'@\w+', ''),
            (r'[^\w\s#]', ' '),
            (r'\s+', ' ')
        ]

        try:
            for pattern, repl in patterns:
                text = re.sub(pattern, repl, text)
            return text.strip()
        except Exception:
            return ""

    def detect_dark_patterns(self, texts, threshold=0.85):
        """Process batch of texts with progress tracking"""
        if not texts:
            return pd.DataFrame()

        results = []
        for text in tqdm(texts, desc="Analyzing Texts"):
            try:
                clean_text = self._clean_text(text)
                if not clean_text or len(clean_text) < 5:
                    continue

                # Initialize prediction variable
                pred = None

                # Basic pattern matching
                is_dark = any(
                    phrase in clean_text.lower()
                    for phrase in ['forwarded', 'via dm', 'private share', 'confidential']
                )

                # NLP classification if no obvious pattern
                if not is_dark and self.dark_detector:
                    pred = self.dark_detector(clean_text[:512])[0]
                    is_dark = pred['label'] == 'POSITIVE' and pred['score'] >= threshold

                results.append({
                    'original': text[:200] + '...' if len(text) > 200 else text,
                    'clean': clean_text[:200] + '...' if len(clean_text) > 200 else clean_text,
                    'is_dark': is_dark,
                    'score': pred['score'] if pred is not None else None  # Safe reference
                })

            except Exception as e:
                logging.warning(f"Failed to process text: {str(e)}")
                continue

        return pd.DataFrame(results)
        results = []
        for text in tqdm(texts, desc="Analyzing Texts"):
            try:
                clean_text = self._clean_text(text)
                if not clean_text or len(clean_text) < 5:
                    continue

                is_dark = any(
                    phrase in clean_text.lower()
                    for phrase in ['forwarded', 'via dm', 'private share', 'confidential']
                )

                if not is_dark and self.dark_detector:
                    pred = self.dark_detector(clean_text[:512])[0]
                    is_dark = pred['label'] == 'POSITIVE' and pred['score'] >= threshold

                results.append({
                    'original': text[:200] + '...' if len(text) > 200 else text,
                    'clean': clean_text[:200] + '...' if len(clean_text) > 200 else clean_text,
                    'is_dark': is_dark,
                    'score': pred['score'] if not is_dark and 'pred' in locals() else None
                })

            except Exception as e:
                logging.warning(f"Failed to process text: {str(e)}")
                continue

        return pd.DataFrame(results)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_resources()
        logging.info("Resources cleaned up")


def configure_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ehdsaf.log'),
            logging.StreamHandler()
        ]
    )


def main() -> object:
    configure_logging()

    try:
        with EHDSAFProcessor(headless=False) as processor:
            test_url = "https://example.com"  # Replace with target URL
            logging.info(f"Starting analysis of {test_url}")

            content = processor.scrape_dark_content(test_url)
            if not content:
                logging.warning("No content scraped - check URL or network settings")
                return

            logging.info(f"Found {len(content)} content fragments")
            results = processor.detect_dark_patterns(content[:10])  # Analyze first 10 samples

            if not results.empty:
                print("\n=== Detection Results ===")
                print(results[['original', 'is_dark']].to_markdown(index=False))

                dark_count = results['is_dark'].sum()
                logging.info(f"Detection complete: {dark_count} dark patterns found")
            else:
                logging.warning("No valid content to analyze")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
