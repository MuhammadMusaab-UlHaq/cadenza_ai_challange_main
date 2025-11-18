import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load your internal validation predictions
print("Loading validation predictions...")
val_df = pd.read_csv('validation_set_test.csv')
print(f"Loaded {len(val_df)} predictions")

# Extract predicted scores (convert from 0-100 to 0-1)
val_df['predicted_score'] = val_df['intelligibility_score of whisper large'] / 100.0

# Load ground truth from training metadata
print("Loading ground truth...")
with open('data/cadenza_data_train/metadata/train_metadata.json', 'r') as f:
    metadata = json.load(f)

truth_df = pd.DataFrame(metadata)[['signal', 'correctness']]
truth_df.rename(columns={'signal': 'signal_id', 'correctness': 'actual_score'}, inplace=True)

# Merge predictions with ground truth
print("Merging with ground truth...")
merged_df = pd.merge(val_df[['signal_id', 'predicted_score']], truth_df, on='signal_id')
print(f"Successfully matched {len(merged_df)} samples")

# Extract values
y_pred = merged_df['predicted_score'].values
y_true = merged_df['actual_score'].values

# Calculate metrics
rmse = np.sqrt(np.mean((y_pred - y_true)**2))
ncc, _ = pearsonr(y_pred, y_true)

print(f"\nMetrics:")
print(f"RMSE: {rmse:.4f}")
print(f"NCC (Pearson's r): {ncc:.4f}")

# Create scatter plot
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize=(6, 6))

# Scatter plot
ax.scatter(y_pred, y_true, alpha=0.3, edgecolors='k', s=50, label='Predictions')

# Perfect prediction line
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')

# Formatting
ax.set_xlabel('Predicted Intelligibility Score', fontsize=12)
ax.set_ylabel('Actual Intelligibility Score', fontsize=12)
ax.set_title('Model Performance on Internal Validation Set', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal', 'box')
ax.grid(True)

# Add metrics text
metrics_text = f'RMSE = {rmse:.4f}\nNCC = {ncc:.4f}\nN = {len(merged_df)}'
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax.legend(loc='lower right')

# Save figure
output_file = 'internal_validation_scatter_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nScatter plot saved as {output_file}")

# Save metrics to file
metrics_file = 'internal_validation_metrics.txt'
with open(metrics_file, 'w') as f:
    f.write("Internal Validation Set Performance\n")
    f.write("=" * 50 + "\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"NCC (Pearson's r): {ncc:.4f}\n")
    f.write(f"Number of samples: {len(merged_df)}\n")
    f.write(f"Mean predicted: {y_pred.mean():.4f}\n")
    f.write(f"Mean actual: {y_true.mean():.4f}\n")
    f.write(f"Std predicted: {y_pred.std():.4f}\n")
    f.write(f"Std actual: {y_true.std():.4f}\n")
print(f"Metrics saved to {metrics_file}")

# Save merged data for future reference
merged_df.to_csv('internal_validation_with_ground_truth.csv', index=False)
print(f"Merged data saved to internal_validation_with_ground_truth.csv")

plt.show()
