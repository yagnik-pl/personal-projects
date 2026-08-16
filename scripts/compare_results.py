import argparse
import json
import csv
import os

def format_table(results):
    md = "| Method | Recall@10 | MRR | nDCG@10 | Avg Layers | Compute % | Latency (ms) | Quality Drop % |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    for r in results:
        md += f"| {r['Method']} | {r['Recall@10']:.4f} | {r['MRR']:.4f} | {r['nDCG@10']:.4f} | {r.get('Avg_Layers', '-'):.2f} | {r['Compute_%']:.1f}% | {r.get('Latency_ms', 0):.2f} | {r.get('Quality_Drop_%', 0):.2f}% |\n"
    return md

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/tables")
    args = parser.parse_args()

    results = []
    
    baseline_path = os.path.join(args.results_dir, "baseline_results.json")
    baseline_ndcg = None
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            b_data = json.load(f)
            # Assuming baseline_results is a dict or single-element list
            if isinstance(b_data, list): b_data = b_data[0]
            baseline_ndcg = b_data.get("NDCG@10", b_data.get("nDCG@10", 0))
            results.append({
                "Method": "Baseline",
                "Recall@10": b_data.get("Recall@10", 0),
                "MRR": b_data.get("MRR", 0),
                "nDCG@10": baseline_ndcg,
                "Avg_Layers": 12, # Hardcoded assumption for baseline
                "Compute_%": 100.0,
                "Latency_ms": b_data.get("latency_ms", 0),
                "Quality_Drop_%": 0.0
            })
            
    fixed_path = os.path.join(args.results_dir, "fixed_depth_results.json")
    if os.path.exists(fixed_path):
        with open(fixed_path, "r") as f:
            f_data = json.load(f)
            for d in f_data:
                qdrop = 0.0
                if baseline_ndcg:
                    qdrop = (1.0 - (d["nDCG@10"] / baseline_ndcg)) * 100
                results.append({
                    "Method": f"Fixed (L{d['layer']})",
                    "Recall@10": d["Recall@10"],
                    "MRR": d["MRR"],
                    "nDCG@10": d["nDCG@10"],
                    "Avg_Layers": float(d["layer"]),
                    "Compute_%": d["compute_pct"],
                    "Latency_ms": d.get("median_latency_ms", 0),
                    "Quality_Drop_%": qdrop
                })

    adaptive_path = os.path.join(args.results_dir, "adaptive_results.json")
    if os.path.exists(adaptive_path):
        with open(adaptive_path, "r") as f:
            a_data = json.load(f)
            for d in a_data:
                qdrop = 0.0
                if baseline_ndcg:
                    qdrop = (1.0 - (d["nDCG@10"] / baseline_ndcg)) * 100
                results.append({
                    "Method": f"Adaptive ({d['threshold']})",
                    "Recall@10": d["Recall@10"],
                    "MRR": d["MRR"],
                    "nDCG@10": d["nDCG@10"],
                    "Avg_Layers": d["avg_exit_layer"],
                    "Compute_%": d["compute_pct"],
                    "Latency_ms": d.get("median_latency_ms", 0),
                    "Quality_Drop_%": qdrop
                })

    if not results:
        print("No results found.")
        return

    # Sort and find Pareto optimal
    results.sort(key=lambda x: x["Compute_%"])
    
    # Save
    with open(os.path.join(args.results_dir, "comparison_table.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(args.results_dir, "comparison_table.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    md_str = format_table(results)
    with open(os.path.join(args.results_dir, "comparison_table.md"), "w") as f:
        f.write(md_str)
        
    print(md_str)

if __name__ == "__main__":
    main()
