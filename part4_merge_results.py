import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*80)
print("PART 4: Merge All Results")
print("="*80)

files_needed = [
    'part1_mnist_results.csv',
    'part2_fashionmnist_results.csv',
    'part3_svm_results.csv',
    'part3_cpu_gpu_results.csv'
]

for f in files_needed:
    if not os.path.exists(f):
        print(f"WARNING: {f} not found!")

df_mnist = pd.read_csv('part1_mnist_results.csv')
df_fmnist = pd.read_csv('part2_fashionmnist_results.csv')
df_svm = pd.read_csv('part3_svm_results.csv')
df_q2 = pd.read_csv('part3_cpu_gpu_results.csv')

df_q1a = pd.concat([df_mnist, df_fmnist], ignore_index=True)
print(f"\nTotal Q1(a) experiments: {len(df_q1a)}")

# ==================== Q1(a) Results ====================
print("\n" + "="*80)
print("Q1(a) DEEP LEARNING RESULTS")
print("="*80)

print("\n--- MNIST Results (Best per config) ---")
mnist_pivot = df_mnist.groupby(['Batch Size', 'Optimizer', 'Learning Rate', 'Model'])['Test Accuracy (%)'].max().unstack()
print(mnist_pivot.to_string())

print("\n--- FashionMNIST Results (Best per config) ---")
fmnist_pivot = df_fmnist.groupby(['Batch Size', 'Optimizer', 'Learning Rate', 'Model'])['Test Accuracy (%)'].max().unstack()
print(fmnist_pivot.to_string())

print("\n--- Effect of Epochs ---")
epochs_effect = df_q1a.groupby(['Dataset', 'Model', 'Epochs']).agg({
    'Test Accuracy (%)': 'mean',
    'Train Time (ms)': 'mean'
}).round(2)
print(epochs_effect.to_string())

print("\n--- Effect of pin_memory ---")
pin_effect = df_q1a.groupby(['Dataset', 'Model', 'pin_memory']).agg({
    'Test Accuracy (%)': 'mean',
    'Train Time (ms)': 'mean'
}).round(2)
print(pin_effect.to_string())

# ==================== Q1(b) SVM Results ====================
print("\n" + "="*80)
print("Q1(b) SVM RESULTS")
print("="*80)
print(df_svm.to_string())

# ==================== Q2 CPU vs GPU ====================
print("\n" + "="*80)
print("Q2 CPU vs GPU RESULTS")
print("="*80)
print(df_q2.to_string())

print("\n--- Accuracy Comparison ---")
acc_pivot = df_q2.pivot_table(values='Test Accuracy (%)', index=['Compute', 'Optimizer'], columns='Model')
print(acc_pivot.to_string())

print("\n--- Train Time Comparison (ms) ---")
time_pivot = df_q2.pivot_table(values='Train Time (ms)', index=['Compute', 'Optimizer'], columns='Model')
print(time_pivot.to_string())

print("\n--- FLOPs per Model ---")
flops_df = df_q2[['Model', 'FLOPs']].drop_duplicates()
print(flops_df.to_string(index=False))

# ==================== Final Summary ====================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\n[Q1(a) Deep Learning]")
print(f"  Total experiments: {len(df_q1a)}")
print(f"  Best MNIST accuracy: {df_mnist['Test Accuracy (%)'].max():.2f}%")
print(f"  Best FashionMNIST accuracy: {df_fmnist['Test Accuracy (%)'].max():.2f}%")

print("\n[Q1(b) SVM]")
print(f"  Total experiments: {len(df_svm)}")
print(f"  Best MNIST SVM: {df_svm[df_svm['Dataset']=='MNIST']['Test Accuracy (%)'].max():.2f}%")
print(f"  Best FashionMNIST SVM: {df_svm[df_svm['Dataset']=='FashionMNIST']['Test Accuracy (%)'].max():.2f}%")

print("\n[Q2 CPU vs GPU]")
cpu_res = df_q2[df_q2['Compute'] == 'CPU']
gpu_res = df_q2[df_q2['Compute'] == 'GPU']
if len(cpu_res) > 0:
    print(f"  Avg CPU Train Time: {cpu_res['Train Time (ms)'].mean():.2f}ms")
if len(gpu_res) > 0:
    print(f"  Avg GPU Train Time: {gpu_res['Train Time (ms)'].mean():.2f}ms")
    if len(cpu_res) > 0:
        speedup = cpu_res['Train Time (ms)'].mean() / gpu_res['Train Time (ms)'].mean()
        print(f"  GPU Speedup: {speedup:.2f}x")

# ==================== Save Combined Results ====================
df_q1a.to_csv('final_q1a_results.csv', index=False)
df_svm.to_csv('final_q1b_svm_results.csv', index=False)
df_q2.to_csv('final_q2_cpu_gpu_results.csv', index=False)

# ==================== Generate Plots ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MNIST accuracy by config
ax1 = axes[0, 0]
mnist_best = df_mnist.groupby(['Optimizer', 'Learning Rate', 'Model'])['Test Accuracy (%)'].max().unstack()
mnist_best.plot(kind='bar', ax=ax1)
ax1.set_title('MNIST: Test Accuracy by Config')
ax1.set_ylabel('Accuracy (%)')
ax1.legend(title='Model')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, axis='y')

# Plot 2: FashionMNIST accuracy by config
ax2 = axes[0, 1]
fmnist_best = df_fmnist.groupby(['Optimizer', 'Learning Rate', 'Model'])['Test Accuracy (%)'].max().unstack()
fmnist_best.plot(kind='bar', ax=ax2)
ax2.set_title('FashionMNIST: Test Accuracy by Config')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(title='Model')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, axis='y')

# Plot 3: SVM Results
ax3 = axes[1, 0]
svm_pivot = df_svm.pivot_table(values='Test Accuracy (%)', index='Kernel', columns='Dataset')
svm_pivot.plot(kind='bar', ax=ax3)
ax3.set_title('SVM: Test Accuracy by Kernel')
ax3.set_ylabel('Accuracy (%)')
ax3.legend(title='Dataset')
ax3.tick_params(axis='x', rotation=0)
ax3.grid(True, axis='y')

# Plot 4: CPU vs GPU
ax4 = axes[1, 1]
if len(df_q2) > 0:
    q2_pivot = df_q2.pivot_table(values='Train Time (ms)', index='Model', columns='Compute')
    q2_pivot.plot(kind='bar', ax=ax4)
    ax4.set_title('Q2: Train Time CPU vs GPU')
    ax4.set_ylabel('Train Time (ms)')
    ax4.legend(title='Compute')
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(True, axis='y')

plt.tight_layout()
plt.savefig('final_combined_results.png', dpi=150)
plt.close()

print("\n" + "="*80)
print("FILES SAVED")
print("="*80)
print("  - final_q1a_results.csv")
print("  - final_q1b_svm_results.csv")
print("  - final_q2_cpu_gpu_results.csv")
print("  - final_combined_results.png")
