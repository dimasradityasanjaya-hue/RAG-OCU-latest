"""
Visualize Evaluation Results
Generate charts and graphs from evaluation data
"""
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_latest_evaluation(results_dir: str = "evaluation_results") -> dict:
    """Load the most recent evaluation JSON file"""
    json_files = [
        f for f in os.listdir(results_dir)
        if f.startswith("evaluation_") and f.endswith(".json")
    ]
    
    if not json_files:
        raise FileNotFoundError("No evaluation results found in evaluation_results/")
    
    # Get latest file
    latest_file = sorted(json_files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    print(f"Loading evaluation results from: {latest_file}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_overall_metrics(data: dict, output_dir: str = "evaluation_results"):
    """Create bar chart for overall metrics"""
    overall = data['overall_metrics']
    
    # Select key metrics to display
    metrics_to_plot = {
        'Exact Match': overall['exact_match_mean'] * 100,
        'Word Accuracy': overall['word_accuracy_mean'] * 100,
        'BLEU-1': overall['bleu_1_mean'] * 100,
        'BLEU-4': overall['bleu_4_mean'] * 100,
        'Semantic Similarity': overall['semantic_similarity_mean'] * 100
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
    bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Evaluation Metrics - RAG Translation System', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metrics_overall.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def plot_metrics_by_type(data: dict, output_dir: str = "evaluation_results"):
    """Create grouped bar chart for metrics by type"""
    metrics_by_type = data['metrics_by_type']
    
    if not metrics_by_type:
        print("No metrics by type to plot")
        return
    
    types = list(metrics_by_type.keys())
    
    # Metrics to compare
    exact_match = [metrics_by_type[t]['exact_match_mean'] * 100 for t in types]
    word_acc = [metrics_by_type[t]['word_accuracy_mean'] * 100 for t in types]
    bleu4 = [metrics_by_type[t]['bleu_4_mean'] * 100 for t in types]
    semantic = [metrics_by_type[t]['semantic_similarity_mean'] * 100 for t in types]
    
    x = np.arange(len(types))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - 1.5*width, exact_match, width, label='Exact Match', color='#4CAF50', alpha=0.8)
    ax.bar(x - 0.5*width, word_acc, width, label='Word Accuracy', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.5*width, bleu4, width, label='BLEU-4', color='#FF9800', alpha=0.8)
    ax.bar(x + 1.5*width, semantic, width, label='Semantic Sim', color='#9C27B0', alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evaluation Metrics by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metrics_by_type.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def plot_bleu_scores(data: dict, output_dir: str = "evaluation_results"):
    """Create line chart for BLEU scores (1-4 gram)"""
    overall = data['overall_metrics']
    
    bleu_scores = {
        'BLEU-1': overall['bleu_1_mean'],
        'BLEU-2': overall['bleu_2_mean'],
        'BLEU-3': overall['bleu_3_mean'],
        'BLEU-4': overall['bleu_4_mean']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = list(range(1, 5))
    y = list(bleu_scores.values())
    
    ax.plot(x, y, marker='o', linewidth=2, markersize=10, color='#2196F3')
    
    # Add value labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi + 0.03, f'{yi:.3f}', ha='center', fontweight='bold')
    
    ax.set_xlabel('N-gram Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax.set_title('BLEU Scores by N-gram Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['1-gram', '2-gram', '3-gram', '4-gram'])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'bleu_scores.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def plot_retrieval_quality(data: dict, output_dir: str = "evaluation_results"):
    """Create chart for retrieval quality metrics"""
    test_cases = data['test_cases']
    
    # Extract retrieval similarities
    retrieval_sims = []
    types = []
    
    for tc in test_cases:
        if 'retrieval_metrics' in tc and tc.get('prediction'):
            retrieval_sims.append(tc['retrieval_metrics']['avg_similarity'])
            types.append(tc['type'])
    
    if not retrieval_sims:
        print("No retrieval data to plot")
        return
    
    # Group by type
    type_groups = {}
    for t, sim in zip(types, retrieval_sims):
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(sim)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    positions = []
    data_to_plot = []
    labels = []
    
    for i, (t, sims) in enumerate(sorted(type_groups.items())):
        positions.append(i + 1)
        data_to_plot.append(sims)
        labels.append(t.capitalize())
    
    bp = ax.boxplot(data_to_plot, positions=positions, labels=labels, 
                     patch_artist=True, widths=0.6)
    
    # Color boxes
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retrieval Similarity Score', fontsize=12, fontweight='bold')
    ax.set_title('Retrieval Quality by Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'retrieval_quality.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def plot_confusion_matrix(data: dict, output_dir: str = "evaluation_results"):
    """Create heatmap showing exact match vs not exact match by type"""
    test_cases = data['test_cases']
    
    # Count exact matches by type
    type_counts = {}
    for tc in test_cases:
        if tc.get('prediction'):
            t = tc['type']
            if t not in type_counts:
                type_counts[t] = {'exact': 0, 'not_exact': 0, 'total': 0}
            
            type_counts[t]['total'] += 1
            if tc['translation_metrics']['exact_match'] == 1.0:
                type_counts[t]['exact'] += 1
            else:
                type_counts[t]['not_exact'] += 1
    
    if not type_counts:
        print("No data for confusion matrix")
        return
    
    types = sorted(type_counts.keys())
    exact_counts = [type_counts[t]['exact'] for t in types]
    not_exact_counts = [type_counts[t]['not_exact'] for t in types]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, exact_counts, width, label='Exact Match', 
                   color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x + width/2, not_exact_counts, width, label='Not Exact Match', 
                   color='#F44336', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Exact Match Distribution by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'exact_match_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def generate_summary_report(data: dict, output_dir: str = "evaluation_results"):
    """Generate a visual summary report combining key info"""
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('RAG Translation System - Evaluation Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Subtitle with timestamp
    timestamp = data.get('timestamp', 'N/A')
    fig.text(0.5, 0.94, f'Evaluation Date: {timestamp}', 
             ha='center', fontsize=10, style='italic')
    
    # Grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Overall metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    overall = data['overall_metrics']
    metrics = {
        'Exact\nMatch': overall['exact_match_mean'] * 100,
        'Word\nAccuracy': overall['word_accuracy_mean'] * 100,
        'BLEU-4': overall['bleu_4_mean'] * 100,
        'Semantic\nSimilarity': overall['semantic_similarity_mean'] * 100
    }
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    bars = ax1.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Overall Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', labelsize=8)
    
    # 2. Test distribution (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if data.get('metrics_by_type'):
        types = list(data['metrics_by_type'].keys())
        counts = []
        for tc_type in types:
            count = sum(1 for tc in data['test_cases'] 
                       if tc.get('type') == tc_type and tc.get('prediction'))
            counts.append(count)
        
        colors = ['#4CAF50', '#2196F3', '#FF9800'][:len(types)]
        wedges, texts, autotexts = ax2.pie(counts, labels=[t.capitalize() for t in types], 
                                            autopct='%1.0f%%', colors=colors, 
                                            startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Test Distribution', fontsize=12, fontweight='bold')
    
    # 3. Metrics by type (middle row, full width)
    ax3 = fig.add_subplot(gs[1, :])
    if data.get('metrics_by_type'):
        types = list(data['metrics_by_type'].keys())
        exact_match = [data['metrics_by_type'][t]['exact_match_mean'] * 100 for t in types]
        semantic = [data['metrics_by_type'][t]['semantic_similarity_mean'] * 100 for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        
        ax3.bar(x - width/2, exact_match, width, label='Exact Match', 
               color='#4CAF50', alpha=0.8)
        ax3.bar(x + width/2, semantic, width, label='Semantic Similarity', 
               color='#9C27B0', alpha=0.8)
        
        ax3.set_xlabel('Category', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
        ax3.set_title('Performance by Category', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([t.capitalize() for t in types])
        ax3.legend()
        ax3.set_ylim(0, 105)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Key stats (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    stats_text = f"""
    ðŸ“Š KEY STATISTICS
    
    Total Test Cases: {data['num_test_cases']}
    Overall Exact Match: {overall['exact_match_mean']*100:.1f}%
    Overall Word Accuracy: {overall['word_accuracy_mean']*100:.1f}%
    BLEU Average: {overall['bleu_avg_mean']:.3f}
    Semantic Similarity: {overall['semantic_similarity_mean']:.3f}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 5. Performance indicator (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Determine overall grade
    avg_score = (overall['exact_match_mean'] + overall['semantic_similarity_mean']) / 2
    if avg_score >= 0.95:
        grade = "EXCELLENT"
        color = '#4CAF50'
        emoji = "ðŸŒŸ"
    elif avg_score >= 0.85:
        grade = "VERY GOOD"
        color = '#8BC34A'
        emoji = "âœ…"
    elif avg_score >= 0.75:
        grade = "GOOD"
        color = '#FFC107'
        emoji = "ðŸ‘"
    else:
        grade = "NEEDS IMPROVEMENT"
        color = '#FF9800'
        emoji = "âš ï¸"
    
    grade_text = f"{emoji}\n\n{grade}\n\n{avg_score*100:.1f}%"
    ax5.text(0.5, 0.5, grade_text, fontsize=20, fontweight='bold',
            ha='center', va='center', color=color,
            bbox=dict(boxstyle='round,pad=1', facecolor=color, alpha=0.2, edgecolor=color, linewidth=3))
    
    output_file = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"âœ“ Saved: {output_file}")
    plt.close()


def main():
    """Main visualization function"""
    print("\n" + "="*70)
    print("GENERATING EVALUATION VISUALIZATIONS".center(70))
    print("="*70 + "\n")
    
    try:
        # Load latest evaluation data
        data = load_latest_evaluation()
        print(f"\nâœ“ Loaded {data['num_test_cases']} test cases\n")
        
        # Generate all visualizations
        print("Creating visualizations...\n")

        plot_overall_metrics(data)
        plot_metrics_by_type(data)
        plot_bleu_scores(data)
        plot_retrieval_quality(data)
        plot_confusion_matrix(data)
        generate_summary_report(data)
        
        print("\n" + "="*70)
        print("VISUALIZATION COMPLETED".center(70))
        print("="*70)
        print("\nâœ“ All charts saved to 'evaluation_results/' folder")
        print("âœ“ Open the PNG files to view the visualizations\n")
        
    except Exception as e:
        print(f"\nâœ— Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
