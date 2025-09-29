import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def load_data(file_path):
    """Load data from Excel file with proper error handling"""
    try:
        print("Loading data...")
        df = pd.read_excel(file_path, engine='openpyxl')

        # Validate required columns
        if 'image_data_grayscale' not in df.columns or 'label' not in df.columns:
            raise ValueError("Missing required columns: 'image_data_grayscale' or 'label'")

        return df
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Please ensure:")
        print("1. The file exists and is not corrupted")
        print("2. The file contains 'image_data_grayscale' and 'label' columns")
        print("3. You have openpyxl installed (pip install openpyxl)")
        return None


def process_image(img_str):
    """Process a single image string to numpy array with memory efficiency"""
    try:
        if pd.isna(img_str):
            return None

        # Robust cleaning and conversion
        cleaned = (img_str.replace('[', ' ')
                   .replace(']', ' ')
                   .replace(',', ' ')
                   .replace('\n', ' '))

        nums = []
        for x in cleaned.split():
            try:
                nums.append(float(x))
            except ValueError:
                continue

        # Ensure correct size (299x299 = 89401 elements)
        target_size = 299 * 299
        if len(nums) < target_size:
            nums.extend([0.0] * (target_size - len(nums)))
        elif len(nums) > target_size:
            nums = nums[:target_size]

        return np.array(nums, dtype=np.float32).reshape(299, 299)  # float32 saves memory
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return None


def process_in_batches(df, batch_size=500):
    """Process images in batches to avoid memory overload"""
    features = []
    labels = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i + batch_size]

        # Process images in current batch
        batch_images = []
        batch_labels = []

        for _, row in batch.iterrows():
            img = process_image(row['image_data_grayscale'])
            if img is not None:
                batch_images.append(img)
                batch_labels.append(row['label'])

        # Extract features for current batch
        if batch_images:
            batch_features = []
            for img in batch_images:
                try:
                    fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), channel_axis=None)
                    batch_features.append(fd)
                except:
                    # Fallback zero vector if feature extraction fails
                    batch_features.append(np.zeros(288, dtype=np.float32))

            # Store results
            features.extend(batch_features)
            labels.extend(batch_labels)

        # Memory cleanup
        del batch_images, batch_features
        print(f"Processed batch {i // batch_size + 1} - Total samples: {len(labels)}")

    return np.array(features), np.array(labels)


def run_analysis():
    print("COVID-19 Image Analysis Pipeline")
    print("=" * 50)

    # Configuration
    data_file = 'COVID19_DatasetAnalysis.xlsx'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    df = load_data(data_file)
    if df is None:
        return

    # 2. Process data in batches
    X, y = process_in_batches(df)

    if len(X) == 0:
        print("Error: No valid images processed")
        return

    print(f"\nSuccessfully processed {len(X)} images")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Train model (with reduced parameters for memory efficiency)
    print("\nTraining classifier...")
    clf = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100 to save memory
        max_depth=10,  # Limit tree depth
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save outputs
    # Save model and scaler
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, 'classification_report.csv'))

    # Save feature matrix and labels
    np.save(os.path.join(output_dir, 'features.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), y)

    print("\nAnalysis complete! Results saved in 'output' folder")
    print("Files generated:")
    print("- output/model.pkl (trained model)")
    print("- output/confusion_matrix.png")
    print("- output/classification_report.csv")
    print("- output/features.npy")
    print("- output/labels.npy")


if __name__ == "__main__":
    run_analysis()