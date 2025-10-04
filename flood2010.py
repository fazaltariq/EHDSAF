import kagglehub

# Download latest version
path = kagglehub.dataset_download("foziashareen/floods-pakistan")

print("Path to dataset files:", path)