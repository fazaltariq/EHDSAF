import os
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
import pd
import seaborn as sns


class Visualizer:
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_visuals(self, df):
        """Generate all visualizations in JPG format"""
        try:
            # 1. Detection Distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(x='is_dark_pattern', data=df, palette=['#4CAF50', '#F44336'])
            plt.title('Dark Pattern Detection Distribution')
            plt.savefig(f'{self.output_dir}/detection_distribution.jpg',
                        format='jpg',
                        dpi=300,
                        quality=90)
            plt.close()

            # 2. Confidence Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df[df['is_dark_pattern']],
                         x='confidence',
                         bins=10,
                         color='#FF5722')
            plt.title('Confidence Score Distribution')
            plt.savefig(f'{self.output_dir}/confidence_distribution.jpg',
                        format='jpg',
                        dpi=300,
                        quality=90)
            plt.close()

            # 3. Timeline Analysis
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            plt.figure(figsize=(12, 6))
            df.set_index('timestamp')['is_dark_pattern'].resample('D').sum().plot()
            plt.title('Daily Detection Trends')
            plt.savefig(f'{self.output_dir}/daily_trends.jpg',
                        format='jpg',
                        dpi=300,
                        quality=90)
            plt.close()

            # Create zip archive
            self._create_zip_archive()

            return True
        except Exception as e:
            import logging
            logging.error(f"Visualization failed: {str(e)}")
            return False

    def _create_zip_archive(self):
        """Package visuals into downloadable zip"""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(self.output_dir):
                if file.endswith('.jpg'):
                    zipf.write(
                        os.path.join(self.output_dir, file),
                        arcname=file
                    )

        with open(f'{self.output_dir}/visualizations.zip', 'wb') as f:
            f.write(zip_buffer.getvalue())

    def get_download_links(self):
        """Return HTML download links for easy access"""
        links = []
        for file in os.listdir(self.output_dir):
            if file.endswith(('.jpg', '.zip')):
                links.append(
                    f'<a href="visualizations/{file}" download>{file}</a>'
                )
        return '<br>'.join(links)


# In your pipeline's run() method, add:
def run(self, source_url, results_df=None):
    # ... existing processing code ...

    # Generate visuals
    visualizer = Visualizer()
    if visualizer.generate_visuals(results_df):
        print("\nDownload visualizations:")
        print(f"1. {os.path.abspath('visualizations/detection_distribution.jpg')}")
        print(f"2. {os.path.abspath('visualizations/confidence_distribution.jpg')}")
        print(f"3. {os.path.abspath('visualizations/daily_trends.jpg')}")
        print(f"4. {os.path.abspath('visualizations/visualizations.zip')} (All files)")

        # For web interfaces
        print("\nHTML download links:")
        print(visualizer.get_download_links())