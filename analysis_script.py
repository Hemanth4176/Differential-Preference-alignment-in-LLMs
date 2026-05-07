"""
Analysis script for GRADE vs GRPO vs Hybrid training results.
Generates figures from results in grade_only/, grpo_only/, hybrid/ directories.
"""
import json, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.figsize': (5.5, 4), 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {'grade_only': '#2ecc71', 'grpo_only': '#e74c3c', 'hybrid': '#f39c12'}
LABELS = {'grade_only': 'GRADE', 'grpo_only': 'GRPO', 'hybrid': 'Hybrid'}


def smooth(data, window=20):
    if not data or len(data) < 2:
        return data
    alpha = 2 / (window + 1)
    s = [data[0]]
    for x in data[1:]:
        s.append(alpha * x + (1 - alpha) * s[-1])
    return s


def load_results(results_dir: Path) -> dict:
    """Load results from nested directory structure (e.g. grade_only/grade_only/results.json)."""
    results = {}
    for method in ['grade_only', 'grpo_only', 'hybrid']:
        # Try nested path first, then flat
        for p in [results_dir / method / method / "results.json",
                  results_dir / method / "results.json"]:
            if p.exists():
                with open(p) as f:
                    results[method] = json.load(f)
                # Also load test_results.json if separate
                tr = p.parent / "test_results.json"
                if tr.exists():
                    with open(tr) as f:
                        results[method]['test_results'] = json.load(f)
                print(f"  Loaded {method} from {p} ({len(results[method].get('reward', []))} steps)")
                break
    return results


def fig1_training_reward(results, out):
    """Training reward (exact accuracy) over steps."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        r = d.get('reward', d.get('exact_accuracy', []))
        if r:
            ax.plot(smooth(r), color=COLORS[m], label=LABELS[m], linewidth=2, alpha=0.9)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('Reward (Exact Accuracy)')
    ax.set_title('Training Reward Over Steps'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig1_training_reward.png'); fig.savefig(out/'fig1_training_reward.pdf'); plt.close()
    print("  Saved fig1_training_reward")


def fig2_training_loss(results, out):
    """Training loss over steps."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        loss = d.get('loss', [])
        if loss:
            ax.plot(smooth(loss), color=COLORS[m], label=LABELS[m], linewidth=2, alpha=0.9)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Steps'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig2_training_loss.png'); fig.savefig(out/'fig2_training_loss.pdf'); plt.close()
    print("  Saved fig2_training_loss")


def fig3_kl_divergence(results, out):
    """KL divergence from reference model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        kl = d.get('kl', [])
        if kl:
            ax.plot(smooth(kl), color=COLORS[m], label=LABELS[m], linewidth=2, alpha=0.9)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence from Reference Model'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig3_kl_divergence.png'); fig.savefig(out/'fig3_kl_divergence.pdf'); plt.close()
    print("  Saved fig3_kl_divergence")


def fig4_proxy_reward(results, out):
    """Proxy reward and grounded proxy reward (GRADE & Hybrid only)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m, d in results.items():
        pr = d.get('proxy_reward', [])
        if pr:
            axes[0].plot(smooth(pr), color=COLORS[m], label=LABELS[m], linewidth=2)
        prg = d.get('proxy_reward_grounded', [])
        if prg:
            axes[1].plot(smooth(prg), color=COLORS[m], label=LABELS[m], linewidth=2)
    axes[0].set_xlabel('Steps'); axes[0].set_ylabel('Proxy Reward')
    axes[0].set_title('(a) Proxy Reward'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Steps'); axes[1].set_ylabel('Grounded Proxy Reward')
    axes[1].set_title('(b) Grounded Proxy Reward'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig4_proxy_reward.png'); fig.savefig(out/'fig4_proxy_reward.pdf'); plt.close()
    print("  Saved fig4_proxy_reward")


def fig5_trust_factor(results, out):
    """Trust factor evolution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        tf = d.get('trust_factor', [])
        if tf:
            ax.plot(tf, color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('Trust Factor')
    ax.set_title('Proxy Verifier Trust Factor'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Trust=0.5')
    fig.tight_layout(); fig.savefig(out/'fig5_trust_factor.png'); fig.savefig(out/'fig5_trust_factor.pdf'); plt.close()
    print("  Saved fig5_trust_factor")


def fig6_temperature(results, out):
    """Temperature annealing schedule."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        tau = d.get('tau', [])
        if tau:
            ax.plot(tau, color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('Temperature (τ)')
    ax.set_title('Gumbel-Softmax Temperature Annealing'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig6_temperature.png'); fig.savefig(out/'fig6_temperature.pdf'); plt.close()
    print("  Saved fig6_temperature")


def fig7_gradient_analysis(results, out):
    """Gradient norm analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    grad_data, gd_names, gd_keys = [], [], []
    for m, d in results.items():
        gm = d.get('grad_norm_mean', [])
        if gm:
            axes[0].plot(smooth(gm), color=COLORS[m], label=LABELS[m], linewidth=2)
        gs = d.get('grad_norm_std', [])
        if gs:
            grad_data.append(gs); gd_names.append(LABELS[m]); gd_keys.append(m)
    axes[0].set_xlabel('Steps'); axes[0].set_ylabel('Gradient Norm')
    axes[0].set_title('(a) Gradient Norm Over Training'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if grad_data:
        bp = axes[1].boxplot(grad_data, labels=gd_names, patch_artist=True)
        for patch, mk in zip(bp['boxes'], gd_keys):
            patch.set_facecolor(COLORS.get(mk, 'gray')); patch.set_alpha(0.7)
    axes[1].set_ylabel('Gradient Std Dev'); axes[1].set_title('(b) Gradient Variance'); axes[1].grid(True, alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(out/'fig7_gradient_analysis.png'); fig.savefig(out/'fig7_gradient_analysis.pdf'); plt.close()
    print("  Saved fig7_gradient_analysis")


def fig8_test_accuracy(results, out):
    """Final test accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods, accs, colors = [], [], []
    for m, d in results.items():
        te = d.get('test_eval', {})
        tr = d.get('test_results', {})
        acc = tr.get('accuracy', te.get('mean_reward', None))
        if acc is not None:
            methods.append(LABELS[m]); accs.append(acc * 100); colors.append(COLORS[m])
    if methods:
        x = np.arange(len(methods))
        bars = ax.bar(x, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Final Test Accuracy on GSM8K (400 samples)')
        ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
    fig.tight_layout(); fig.savefig(out/'fig8_test_accuracy.png'); fig.savefig(out/'fig8_test_accuracy.pdf'); plt.close()
    print("  Saved fig8_test_accuracy")


def fig9_running_accuracy(results, out):
    """Running average accuracy over training."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in results.items():
        r = d.get('reward', d.get('exact_accuracy', []))
        if r:
            cumsum = np.cumsum(r)
            running = cumsum / (np.arange(len(r)) + 1)
            ax.plot(running, color=COLORS[m], label=LABELS[m], linewidth=2)
    ax.set_xlabel('Training Steps'); ax.set_ylabel('Cumulative Avg Accuracy')
    ax.set_title('Running Average Training Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig9_running_accuracy.png'); fig.savefig(out/'fig9_running_accuracy.pdf'); plt.close()
    print("  Saved fig9_running_accuracy")


def fig10_combined_dashboard(results, out):
    """Combined 2x3 dashboard of key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # (0,0) Reward
    for m, d in results.items():
        r = d.get('reward', [])
        if r: axes[0,0].plot(smooth(r), color=COLORS[m], label=LABELS[m], linewidth=1.5)
    axes[0,0].set_title('Training Reward'); axes[0,0].legend(fontsize=7); axes[0,0].grid(True, alpha=0.3)
    # (0,1) Loss
    for m, d in results.items():
        l = d.get('loss', [])
        if l: axes[0,1].plot(smooth(l), color=COLORS[m], label=LABELS[m], linewidth=1.5)
    axes[0,1].set_title('Training Loss'); axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.3)
    # (0,2) KL
    for m, d in results.items():
        k = d.get('kl', [])
        if k: axes[0,2].plot(smooth(k), color=COLORS[m], label=LABELS[m], linewidth=1.5)
    axes[0,2].set_title('KL Divergence'); axes[0,2].legend(fontsize=7); axes[0,2].grid(True, alpha=0.3)
    # (1,0) Proxy Reward
    for m, d in results.items():
        pr = d.get('proxy_reward_grounded', [])
        if pr: axes[1,0].plot(smooth(pr), color=COLORS[m], label=LABELS[m], linewidth=1.5)
    axes[1,0].set_title('Grounded Proxy Reward'); axes[1,0].legend(fontsize=7); axes[1,0].grid(True, alpha=0.3)
    # (1,1) Trust Factor
    for m, d in results.items():
        tf = d.get('trust_factor', [])
        if tf: axes[1,1].plot(tf, color=COLORS[m], label=LABELS[m], linewidth=1.5)
    axes[1,1].set_title('Trust Factor'); axes[1,1].legend(fontsize=7); axes[1,1].grid(True, alpha=0.3)
    # (1,2) Test Accuracy Bar
    methods, accs, colors = [], [], []
    for m, d in results.items():
        te = d.get('test_eval', {}); tr = d.get('test_results', {})
        acc = tr.get('accuracy', te.get('mean_reward', None))
        if acc is not None:
            methods.append(LABELS[m]); accs.append(acc*100); colors.append(COLORS[m])
    if methods:
        x = np.arange(len(methods))
        bars = axes[1,2].bar(x, accs, color=colors, alpha=0.85)
        axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(methods)
        for bar, a in zip(bars, accs):
            axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{a:.1f}%', ha='center', fontsize=8)
    axes[1,2].set_title('Test Accuracy (%)'); axes[1,2].set_ylim(0, 100); axes[1,2].grid(True, alpha=0.3, axis='y')
    fig.suptitle('GRADE vs GRPO vs Hybrid — Training Dashboard', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(out/'fig10_dashboard.png'); fig.savefig(out/'fig10_dashboard.pdf'); plt.close()
    print("  Saved fig10_dashboard")


def fig11_grpo_metrics(results, out):
    """GRPO-specific metrics: policy_loss, groups_with_signal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m, d in results.items():
        pl = d.get('policy_loss', [])
        if pl: axes[0].plot(smooth(pl), color=COLORS[m], label=LABELS[m], linewidth=2)
    axes[0].set_xlabel('Steps'); axes[0].set_ylabel('Policy Loss')
    axes[0].set_title('(a) GRPO Policy Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for m, d in results.items():
        gs = d.get('groups_with_signal', [])
        tg = d.get('total_groups', [])
        if gs and tg:
            ratio = [g/t if t > 0 else 0 for g, t in zip(gs, tg)]
            axes[1].plot(smooth(ratio), color=COLORS[m], label=LABELS[m], linewidth=2)
    axes[1].set_xlabel('Steps'); axes[1].set_ylabel('Fraction with Signal')
    axes[1].set_title('(b) Groups with Learning Signal'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out/'fig11_grpo_metrics.png'); fig.savefig(out/'fig11_grpo_metrics.pdf'); plt.close()
    print("  Saved fig11_grpo_metrics")


def fig12_hybrid_decomposition(results, out):
    """Decompose hybrid into GRADE steps (odd) and GRPO steps (even)."""
    if 'hybrid' not in results:
        print("  Skipping fig12 (no hybrid data)"); return
    d = results['hybrid']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Loss: separate GRADE (even idx) vs GRPO (odd idx) steps
    loss = d.get('loss', [])
    grade_loss = [loss[i] for i in range(0, len(loss), 2)]
    grpo_loss = [loss[i] for i in range(1, len(loss), 2)]
    axes[0].plot(smooth(grade_loss), color=COLORS['grade_only'], label='GRADE steps', linewidth=2)
    axes[0].plot(smooth(grpo_loss), color=COLORS['grpo_only'], label='GRPO steps', linewidth=2)
    axes[0].set_title('(a) Hybrid Loss by Step Type'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    # KL
    kl = d.get('kl', [])
    grade_kl = [kl[i] for i in range(0, len(kl), 2)]
    grpo_kl = [kl[i] for i in range(1, len(kl), 2)]
    axes[1].plot(smooth(grade_kl), color=COLORS['grade_only'], label='GRADE steps', linewidth=2)
    axes[1].plot(smooth(grpo_kl), color=COLORS['grpo_only'], label='GRPO steps', linewidth=2)
    axes[1].set_title('(b) Hybrid KL by Step Type'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    # Reward
    rw = d.get('reward', [])
    grade_rw = [rw[i] for i in range(0, len(rw), 2)]
    grpo_rw = [rw[i] for i in range(1, len(rw), 2)]
    axes[2].plot(smooth(grade_rw), color=COLORS['grade_only'], label='GRADE steps', linewidth=2)
    axes[2].plot(smooth(grpo_rw), color=COLORS['grpo_only'], label='GRPO steps', linewidth=2)
    axes[2].set_title('(c) Hybrid Reward by Step Type'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle('Hybrid Mode — GRADE vs GRPO Step Decomposition', fontweight='bold')
    fig.tight_layout(); fig.savefig(out/'fig12_hybrid_decomposition.png'); fig.savefig(out/'fig12_hybrid_decomposition.pdf'); plt.close()
    print("  Saved fig12_hybrid_decomposition")


def statistical_report(results, out):
    """Generate statistical analysis report."""
    lines = ["=" * 60, "STATISTICAL ANALYSIS REPORT", "=" * 60, ""]
    # Test Performance
    lines.append("## Test Performance")
    for m, d in results.items():
        te = d.get('test_eval', {}); tr = d.get('test_results', {})
        acc = tr.get('accuracy', te.get('mean_reward', 0))
        total = tr.get('total', 'N/A'); correct = tr.get('correct', 'N/A')
        lines.append(f"  {LABELS[m]}: {acc*100:.1f}% ({correct}/{total})")
    # Training Stats
    lines.append("\n## Training Statistics (last 20% of steps)")
    for m, d in results.items():
        r = d.get('reward', [])
        if r:
            n = max(1, len(r)//5)
            tail = r[-n:]
            lines.append(f"  {LABELS[m]}: mean={np.mean(tail):.3f}, std={np.std(tail):.3f}, steps={len(r)}")
    # Pairwise t-tests
    methods = list(results.keys())
    if len(methods) >= 2:
        lines.append("\n## Pairwise t-tests (last 20% rewards)")
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                r1 = results[m1].get('reward', []); r2 = results[m2].get('reward', [])
                if r1 and r2:
                    n1, n2 = max(1,len(r1)//5), max(1,len(r2)//5)
                    t, p = stats.ttest_ind(r1[-n1:], r2[-n2:])
                    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
                    lines.append(f"  {LABELS[m1]} vs {LABELS[m2]}: t={t:.3f}, p={p:.4f} {sig}")
    # Gradient stats
    lines.append("\n## Gradient Statistics")
    for m, d in results.items():
        gm = d.get('grad_norm_mean', []); gs = d.get('grad_norm_std', [])
        if gm:
            lines.append(f"  {LABELS[m]}: norm_mean={np.mean(gm):.4f}, norm_std={np.mean(gs):.4f}")
    # KL stats
    lines.append("\n## KL Divergence Statistics")
    for m, d in results.items():
        kl = d.get('kl', [])
        if kl:
            nonzero = [k for k in kl if k != 0]
            if nonzero:
                lines.append(f"  {LABELS[m]}: mean={np.mean(nonzero):.3f}, max={np.max(nonzero):.3f}, min={np.min(nonzero):.3f}")
    report = "\n".join(lines)
    print(report)
    with open(out / "statistical_analysis.txt", "w") as f:
        f.write(report)
    return report


def main(results_dir=None):
    if results_dir is None:
        parser = argparse.ArgumentParser(description="Analyze GRADE/GRPO/Hybrid results")
        parser.add_argument("--results_dir", type=str, default="./results")
        args = parser.parse_args()
        results_dir = args.results_dir

    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    if not results:
        print("No results found! Check directory structure."); return

    print(f"Found: {list(results.keys())}\n")
    print("Generating figures...")
    fig1_training_reward(results, figures_dir)
    fig2_training_loss(results, figures_dir)
    fig3_kl_divergence(results, figures_dir)
    fig4_proxy_reward(results, figures_dir)
    fig5_trust_factor(results, figures_dir)
    fig6_temperature(results, figures_dir)
    fig7_gradient_analysis(results, figures_dir)
    fig8_test_accuracy(results, figures_dir)
    fig9_running_accuracy(results, figures_dir)
    fig10_combined_dashboard(results, figures_dir)
    fig11_grpo_metrics(results, figures_dir)
    fig12_hybrid_decomposition(results, figures_dir)

    print("\nRunning statistical analysis...")
    statistical_report(results, figures_dir)

    print(f"\n{'='*50}")
    print(f"All outputs saved to {figures_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()