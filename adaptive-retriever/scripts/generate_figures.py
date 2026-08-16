import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

def generate_quality_vs_depth(tables_dir, figs_dir):
    fixed_path = os.path.join(tables_dir, "fixed_depth_results.json")
    if not os.path.exists(fixed_path):
        print(f"Skipping quality_vs_depth: {fixed_path} not found")
        return
        
    with open(fixed_path, "r") as f:
        data = json.load(f)
        
    layers = [d["layer"] for d in data]
    ndcg = [d["nDCG@10"] for d in data]
    recall = [d["Recall@10"] for d in data]
    mrr = [d["MRR"] for d in data]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(layers, ndcg, marker='o', label='nDCG@10', linewidth=2)
    ax.plot(layers, recall, marker='s', label='Recall@10', linewidth=2)
    ax.plot(layers, mrr, marker='^', label='MRR', linewidth=2)
    
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Quality vs Layer Depth')
    ax.set_xticks(layers)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "quality_vs_depth.png"), dpi=300)
    plt.savefig(os.path.join(figs_dir, "quality_vs_depth.pdf"))
    plt.close()

def generate_computation_vs_quality(tables_dir, figs_dir):
    comp_path = os.path.join(tables_dir, "comparison_table.json")
    if not os.path.exists(comp_path):
        print(f"Skipping computation_vs_quality: {comp_path} not found")
        return
        
    with open(comp_path, "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for d in data:
        method = d["Method"]
        comp = d["Compute_%"]
        ndcg = d["nDCG@10"]
        
        if "Baseline" in method:
            ax.scatter(comp, ndcg, color='red', marker='*', s=200, label='Full Depth')
        elif "Fixed" in method:
            ax.scatter(comp, ndcg, color='blue', marker='o', alpha=0.5)
        elif "Adaptive" in method:
            ax.scatter(comp, ndcg, color='green', marker='s', s=100)
            # label adaptive points
            if "strict" in method or "medium" in method or "aggressive" in method:
                ax.annotate(method.split(" ")[1], (comp, ndcg), xytext=(5, 5), textcoords='offset points')
                
    # Add dummy legend entries
    ax.scatter([], [], color='blue', marker='o', label='Fixed Depth')
    ax.scatter([], [], color='green', marker='s', label='Adaptive')
    
    ax.set_xlabel('Compute % (Relative to Full Depth)')
    ax.set_ylabel('nDCG@10')
    ax.set_title('Computation vs Retrieval Quality')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "computation_vs_quality.png"), dpi=300)
    plt.savefig(os.path.join(figs_dir, "computation_vs_quality.pdf"))
    plt.close()

def generate_exit_distribution(raw_dir, figs_dir):
    dist_path = os.path.join(raw_dir, "exit_layer_distribution.json")
    if not os.path.exists(dist_path):
        print(f"Skipping exit_distribution: {dist_path} not found")
        return
        
    with open(dist_path, "r") as f:
        data = json.load(f)
        
    if "medium" not in data:
        print("Medium threshold data not found for distribution plot")
        return
        
    counts = np.array(data["medium"])
    layers = np.arange(len(counts))
    
    # Filter out layers with zero counts at the end if any
    mask = counts > 0
    if sum(mask) > 0:
        max_layer = max(layers[mask])
        counts = counts[:max_layer+1]
        layers = layers[:max_layer+1]
    
    # Exclude layer 0 which is empty
    counts = counts[1:]
    layers = layers[1:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(layers, counts, color='skyblue', edgecolor='black')
    
    total = sum(counts)
    mean_layer = sum([l*c for l, c in zip(layers, counts)]) / total
    
    cum_counts = np.cumsum(counts)
    med_idx = np.where(cum_counts >= total/2)[0][0]
    med_layer = layers[med_idx]
    
    ax.axvline(mean_layer, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean_layer:.1f})')
    ax.axvline(med_layer, color='green', linestyle='-', linewidth=2, label=f'Median ({med_layer})')
    
    ax.set_xlabel('Exit Layer')
    ax.set_ylabel('Number of Queries')
    ax.set_title('Exit Layer Distribution (Medium Threshold)')
    ax.set_xticks(layers)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "exit_layer_distribution.png"), dpi=300)
    plt.savefig(os.path.join(figs_dir, "exit_layer_distribution.pdf"))
    plt.close()

def generate_threshold_comparison(tables_dir, figs_dir):
    adaptive_path = os.path.join(tables_dir, "adaptive_results.json")
    if not os.path.exists(adaptive_path):
        print(f"Skipping threshold_comparison: {adaptive_path} not found")
        return
        
    with open(adaptive_path, "r") as f:
        data = json.load(f)
        
    named_data = [d for d in data if d["threshold"] in ["strict", "medium", "aggressive"]]
    if not named_data:
        print("No named thresholds found in adaptive results.")
        return
        
    named_data.sort(key=lambda x: {"aggressive": 1, "medium": 2, "strict": 3}.get(x["threshold"], 4))
    
    labels = [d["threshold"].capitalize() for d in named_data]
    ndcgs = [d["nDCG@10"] for d in named_data]
    computes = [d["compute_pct"] for d in named_data]
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, ndcgs, width, label='nDCG@10', color='royalblue')
    ax1.set_ylabel('nDCG@10', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, computes, width, label='Compute %', color='darkorange')
    ax2.set_ylabel('Compute %', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Comparison of Adaptive Thresholds')
    
    fig.tight_layout()
    plt.savefig(os.path.join(figs_dir, "threshold_comparison.png"), dpi=300)
    plt.savefig(os.path.join(figs_dir, "threshold_comparison.pdf"))
    plt.close()

def main():
    tables_dir = "results/tables"
    figs_dir = "results/figures"
    raw_dir = "results/raw"
    
    os.makedirs(figs_dir, exist_ok=True)
    
    generate_quality_vs_depth(tables_dir, figs_dir)
    generate_computation_vs_quality(tables_dir, figs_dir)
    generate_exit_distribution(raw_dir, figs_dir)
    generate_threshold_comparison(tables_dir, figs_dir)
    
    print(f"Figures generated in {figs_dir}")

if __name__ == "__main__":
    main()
