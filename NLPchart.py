import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np

# Sample data generation (replace with your actual DataFrame)
np.random.seed(42)
data = {
    'original_text': [
        "Forwarded: Confidential documents",
        "Check this https://example.com",
        "Via DM: Private offer",
        "Normal message",
        "EXCLUSIVE ACCESS!!!",
        "Limited time offer",
        "Hello world",
        "Secret promotion",
        "Public announcement",
        "Private share: special deal"
    ],
    'processed_text': [
        "confidential documents",
        "check this",
        "private offer",
        "normal message",
        "exclusive access",
        "limited time offer",
        "hello world",
        "secret promotion",
        "public announcement",
        "private share special deal"
    ],
    'is_dark_pattern': [True, False, True, False, True, True, False, True, False, True],
    'confidence': [1.0, 0.0, 1.0, 0.0, 0.97, 0.92, 0.0, 0.88, 0.0, 1.0],
    'classification_method': ['rule-based', None, 'rule-based', None, 'NLP', 'NLP', None, 'NLP', None, 'rule-based']
}
df = pd.DataFrame(data)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 1. Detection Distribution Pie Chart
plt.figure(figsize=(8, 6))
counts = df['is_dark_pattern'].value_counts()
plt.pie(counts,
        labels=['Normal', 'Dark Pattern'],
        colors=['#4CAF50', '#F44336'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0),
        shadow=True)
plt.title('Dark Pattern Detection Distribution', pad=20, fontsize=14)
plt.tight_layout()
plt.savefig('detection_distribution.png')
plt.show()

# 2. Confidence Score Histogram
plt.figure(figsize=(10, 6))
plt.hist(df[df['is_dark_pattern']]['confidence'],
         bins=10,
         color='#FF5722',
         edgecolor='black',
         alpha=0.8)
plt.xlabel('Confidence Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Confidence Distribution of Dark Patterns', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('confidence_distribution.png')
plt.show()

# 3. Top Dark Pattern Keywords
from collections import Counter
dark_texts = ' '.join(df[df['is_dark_pattern']]['processed_text'])
words = [word for word in dark_texts.lower().split() if len(word) > 3]
word_freq = Counter(words).most_common(10)

plt.figure(figsize=(10, 6))
plt.barh([w[0] for w in word_freq],
         [w[1] for w in word_freq],
         color='#9C27B0',
         edgecolor='black')
plt.xlabel('Frequency', fontsize=12)
plt.title('Top 10 Dark Pattern Keywords', fontsize=14)
plt.gca().invert_yaxis()
plt.savefig('top_keywords.png')
plt.show()

# 4. Word Cloud
plt.figure(figsize=(12, 8))
wordcloud = WordCloud(width=800,
                      height=400,
                      background_color='white',
                      colormap='Reds',
                      max_words=100).generate(dark_texts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Dark Pattern Word Cloud', pad=20, fontsize=14)
plt.savefig('wordcloud.png', bbox_inches='tight')
plt.show()

# 5. Method Comparison
method_counts = df[df['is_dark_pattern']]['classification_method'].value_counts()

plt.figure(figsize=(8, 6))
method_counts.plot(kind='bar',
                  color=['#2196F3', '#FFC107'],
                  edgecolor='black',
                  alpha=0.8)
plt.title('Detection Methods Used', fontsize=14)
plt.xlabel('Method', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.savefig('methods_used.png')
plt.show()