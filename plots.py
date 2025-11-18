
# Generate model architecture diagram and performance scatter plot.

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not installed. Skipping architecture diagram.")

# --- 1. Generate Architecture Diagram ---
if GRAPHVIZ_AVAILABLE:
    try:
        dot = Digraph(comment='Whisper Regression Model Architecture')
        dot.attr(rankdir='TB', splines='ortho')

        node_styles = {
            'input': {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue'},
            'frozen': {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgrey'},
            'trainable': {'shape': 'box', 'style': 'filled', 'fillcolor': 'lightgreen'},
            'process': {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'whitesmoke'},
            'output': {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'gold'}
        }

        # Nodes
        dot.node('audio_in', 'Input Audio Waveform', **node_styles['input'])
        dot.node('hl_in', 'Hearing Loss\n(Categorical)', **node_styles['input'])
        dot.node('whisper_enc', 'Frozen Whisper Encoder\n(large-v3)', **node_styles['frozen'])
        dot.node('mean_pool', 'Mean Pooling', **node_styles['process'])
        dot.node('hl_embed', 'Trainable\nHearing Loss Embedding', **node_styles['trainable'])
        dot.node('concat', 'Concatenate', **node_styles['process'])
        dot.node('reg_head', 'Trainable Regression Head\n(MLP with Dropout)', **node_styles['trainable'])
        dot.node('output_score', 'Predicted Score\n(0 to 1)', **node_styles['output'])

        # Edges
        dot.edge('audio_in', 'whisper_enc')
        dot.edge('whisper_enc', 'mean_pool')
        dot.edge('hl_in', 'hl_embed')
        dot.edge('mean_pool', 'concat')
        dot.edge('hl_embed', 'concat')
        dot.edge('concat', 'reg_head')
        dot.edge('reg_head', 'output_score')

        dot.render('model_architecture', format='png', view=False, cleanup=True)
        print("✓ Architecture diagram saved as model_architecture.png")
    except Exception as e:
        print(f"Warning: Could not generate architecture diagram: {e}")

# --- 2. Load Internal Validation Data ---
print("\nLoading internal validation predictions...")
val_df = pd.read_csv('validation_set_test.csv')
val_df['predicted_score'] = val_df['intelligibility_score of whisper large'] / 100.0

print("Loading ground truth...")
with open('data/cadenza_data_train/metadata/train_metadata.json', 'r') as f:
    metadata = json.load(f)

truth_df = pd.DataFrame(metadata)[['signal', 'correctness']]
truth_df.rename(columns={'signal': 'signal_id', 'correctness': 'actual_score'}, inplace=True)

# Merge
merged_df = pd.merge(val_df[['signal_id', 'predicted_score']], truth_df, on='signal_id')
print(f"✓ Matched {len(merged_df)} samples")

y_pred = merged_df['predicted_score'].values
y_true = merged_df['actual_score'].values

# --- 3. Calculate Metrics ---
rmse = np.sqrt(np.mean((y_pred - y_true)**2))
ncc, _ = pearsonr(y_pred, y_true)

print(f"\nMetrics:")
print(f"  RMSE: {rmse:.4f}")
print(f"  NCC:  {ncc:.4f}")

# --- 4. Create Scatter Plot ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_pred, y_true, alpha=0.3, edgecolors='k', s=50, label='Predictions')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')

ax.set_xlabel('Predicted Intelligibility Score', fontsize=12)
ax.set_ylabel('Actual Intelligibility Score', fontsize=12)
ax.set_title('Model Performance on Internal Validation Set', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal', 'box')
ax.grid(True)

metrics_text = f'RMSE = {rmse:.4f}\nNCC = {ncc:.4f}\nN = {len(merged_df)}'
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax.legend(loc='lower right')

plt.savefig('internal_validation_scatter_plot.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot saved as internal_validation_scatter_plot.png")

# --- 5. Save Metrics ---
with open('internal_validation_metrics.txt', 'w') as f:
    f.write("Internal Validation Set Performance\n")
    f.write("=" * 50 + "\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"NCC (Pearson's r): {ncc:.4f}\n")
    f.write(f"Number of samples: {len(merged_df)}\n")
    f.write(f"Mean predicted: {y_pred.mean():.4f}\n")
    f.write(f"Mean actual: {y_true.mean():.4f}\n")
    f.write(f"Std predicted: {y_pred.std():.4f}\n")
    f.write(f"Std actual: {y_true.std():.4f}\n")
print("Metrics saved to internal_validation_metrics.txt")

merged_df.to_csv('internal_validation_with_ground_truth.csv', index=False)
print("Data saved to internal_validation_with_ground_truth.csv")

plt.show()