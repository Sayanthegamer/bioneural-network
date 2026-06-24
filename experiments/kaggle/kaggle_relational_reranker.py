# ==============================================================================
# SELF-CONTAINED EXPERIMENT 4: RELATIONAL RE-RANKER & DISAMBIGUATION FOR KAGGLE
# ==============================================================================
# Paste this entire script into a single Kaggle Notebook cell.
# It installs dependencies, embeds all architecture code, and runs the full
# two-stage re-ranking evaluation with alias cardinality and Oracle@K telemetry.
# Leverages GPU acceleration if available.
# ==============================================================================

import subprocess
import sys

# Ensure required packages are installed
try:
    import sentence_transformers
except ImportError:
    print("[*] Installing sentence-transformers package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

import os
import csv
import time
import torch
import numpy as np
import random
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==============================================================================
# GPU/CPU DEVICE CONFIGURATION
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Executing on device: {device}")

# ==============================================================================
# 1. REPRODUCIBILITY
# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. FACT GENERATION (same as Experiment 3)
# ==============================================================================

def generate_facts(N):
    facts = []
    random_gen = random.Random(12345)

    rooms = ["vault", "lobby", "lab", "archive", "server", "office", "warehouse",
             "depot", "hangar", "observatory", "basement", "attic", "corridor",
             "lounge", "cafeteria"]
    entrances = ["main door", "side entrance", "emergency exit", "rear gate",
                 "elevator door", "service hatch", "loading dock", "security gate",
                 "roof hatch", "ventilation shaft"]

    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown",
              "black", "white", "gray", "pink", "cyan", "magenta", "teal", "indigo"]
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
                "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron"]
    locations = ["cabinet A", "cabinet B", "drawer 1", "drawer 2", "shelf X",
                 "shelf Y", "safe box", "storage bin", "desk tray", "locker 9"]

    coordinators = ["Sarah", "David", "Emma", "James", "Sophia", "Daniel",
                    "Olivia", "Michael", "Isabella", "William"]
    sessions = ["sync", "review", "planning", "alignment", "retrospective",
                "briefing", "workshop", "interview", "debrief", "consultation"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = ["9 AM", "10 AM", "11 AM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM"]

    for i in range(N):
        num = random_gen.randint(1000, 9999)
        template_type = i % 3

        room = rooms[i % len(rooms)]
        entrance = entrances[(i // len(rooms)) % len(entrances)]

        color = colors[i % len(colors)]
        project = projects[(i // len(colors)) % len(projects)]
        loc = locations[(i // (len(colors) * len(projects))) % len(locations)]

        coord = coordinators[i % len(coordinators)]
        session = sessions[(i // len(coordinators)) % len(sessions)]
        day = days[(i // (len(coordinators) * len(sessions))) % len(days)]
        hour = hours[(i // (len(coordinators) * len(sessions) * len(days))) % len(hours)]

        statements = []
        if template_type == 0:
            statements.append(f"The access code for the {room} {entrance} is {num}.")
            statements.append(f"Entering {room} {entrance} requires keying in {num}.")
            statements.append(f"To unlock the door to {room} {entrance}, type in {num}.")
            statements.append(f"The digit sequence {num} grants entry to {room} {entrance}.")
            statements.append(f"Passcode {num} is configured for access to {room} {entrance}.")
            statements.append(f"You must enter keycode {num} to access {room} {entrance}.")
            statements.append(f"The entry pin for the {room} {entrance} is registered as {num}.")
            statements.append(f"Use password {num} when entering the {room} {entrance}.")
            statements.append(f"To get into {room} {entrance}, type the code {num} on the keypad.")
            statements.append(f"The lock on {room} {entrance} opens with passcode {num}.")
            query = f"Which combination of numbers unlocks the {entrance} of the {room}?"

        elif template_type == 1:
            statements.append(f"The {color} folder for Project {project} is in {loc}.")
            statements.append(f"You can find the Project {project} paperwork of color {color} inside {loc}.")
            statements.append(f"The {loc} contains the {color}-colored documents for Project {project}.")
            statements.append(f"I stored the Project {project} file colored {color} in {loc}.")
            statements.append(f"The {color} Project {project} folder is resting inside {loc}.")
            statements.append(f"Go check {loc} for the {color} folder belonging to Project {project}.")
            statements.append(f"The {color} file for Project {project} is kept in {loc}.")
            statements.append(f"Project {project} has its {color} dossier placed in {loc}.")
            statements.append(f"Inside {loc}, you will locate the {color} folder for Project {project}.")
            statements.append(f"The {color} folder representing Project {project} was left in {loc}.")
            query = f"Where should I look for the documents associated with the {color} Project {project}?"

        else:
            statements.append(f"{coord}'s {session} meeting is {day} at {hour}.")
            statements.append(f"{coord} holds the {session} session on {day} scheduled for {hour}.")
            statements.append(f"The schedule lists {coord}'s {session} on {day} at {hour}.")
            statements.append(f"We meet with {coord} for the {session} on {day} at {hour}.")
            statements.append(f"At {hour} on {day}, {coord} conducts the {session} meeting.")
            statements.append(f"On {day} at {hour}, the {session} session with {coord} begins.")
            statements.append(f"{coord} runs the {session} gathering on {day} starting at {hour}.")
            statements.append(f"{coord}'s schedule includes the {session} on {day} at {hour}.")
            statements.append(f"{coord} will meet us for {session} on {day} at {hour}.")
            statements.append(f"The {session} with {coord} takes place on {day} at {hour}.")
            query = f"What is the day and hour of the {session} session hosted by {coord}?"

        facts.append({
            "label": i,
            "statements": statements,
            "query": query,
            "template_type": template_type
        })
    return facts

# ==============================================================================
# 3. ALIAS CARDINALITY COMPUTATION
# ==============================================================================

def compute_alias_sets(facts):
    """
    For each class, compute how many other classes share the exact same query text.
    Returns a list of alias set sizes (one per class), and a dict mapping
    query text -> list of class labels.
    """
    query_to_labels = defaultdict(list)
    for f in facts:
        query_to_labels[f["query"]].append(f["label"])

    alias_sizes = []
    for f in facts:
        alias_sizes.append(len(query_to_labels[f["query"]]))

    return alias_sizes, dict(query_to_labels)

# ==============================================================================
# 4. TOKEN-MATCHING RE-RANKER
# ==============================================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "that", "this", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "what", "which", "who",
    "whom", "whose"
}

def tokenize_for_overlap(text):
    """Lowercases, strips punctuation, removes stop words."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return set(t for t in tokens if t not in STOP_WORDS)

def token_overlap_score(query_tokens, statement_text):
    """Counts how many non-stop-word tokens in the query appear in the statement."""
    statement_tokens = tokenize_for_overlap(statement_text)
    return len(query_tokens & statement_tokens)

# ==============================================================================
# 5. RE-RANKING EVALUATION CORE
# ==============================================================================

def evaluate_rerankers(facts, statements_emb, queries_emb, cross_encoder_model,
                       K=10, eval_device='cpu'):
    """
    Runs the full re-ranking evaluation pipeline.

    Args:
        facts: list of dicts with 'label', 'statements', 'query', 'template_type'
        statements_emb: [N, 10, D] tensor of statement embeddings
        queries_emb: [N, D] tensor of query embeddings
        cross_encoder_model: loaded CrossEncoder model
        K: number of top candidates to retrieve
        eval_device: torch device string

    Returns:
        A list of per-query result dicts.
    """
    N = len(facts)
    num_exemplars = 10

    # Compute alias cardinality
    alias_sizes, query_to_labels = compute_alias_sets(facts)

    # Compute prototypical centroids
    centroids = statements_emb.mean(dim=1)  # [N, D]

    # Compute L1 distances from each query to each centroid
    dists = torch.cdist(queries_emb.unsqueeze(0), centroids.unsqueeze(0), p=1).squeeze(0)  # [N, N]

    # Get top-K candidates per query
    topK_indices = dists.argsort(dim=1)[:, :K]  # [N, K]

    # Also get top-20 for Oracle@20
    K20 = min(20, N)
    top20_indices = dists.argsort(dim=1)[:, :K20]  # [N, K20]

    results = []

    # Pre-tokenize all queries
    query_token_sets = []
    for f in facts:
        query_token_sets.append(tokenize_for_overlap(f["query"]))

    # Prepare cross-encoder pairs in batch per query
    print(f"    [*] Preparing cross-encoder pairs for N={N}, K={K}...")
    all_ce_pairs = []
    pair_map = []  # (query_idx, candidate_class_idx, exemplar_idx)

    for q_idx in range(N):
        candidates = topK_indices[q_idx].tolist()
        for c_idx in candidates:
            for ex_idx in range(num_exemplars):
                pair_text = (facts[q_idx]["query"], facts[c_idx]["statements"][ex_idx])
                all_ce_pairs.append(pair_text)
                pair_map.append((q_idx, c_idx, ex_idx))

    # Score all pairs with cross-encoder in batches
    print(f"    [*] Scoring {len(all_ce_pairs)} cross-encoder pairs...")
    ce_scores_flat = cross_encoder_model.predict(
        [list(p) for p in all_ce_pairs],
        batch_size=512,
        show_progress_bar=True
    )

    # Organize cross-encoder scores: ce_scores[q_idx][c_idx] = list of 10 scores
    ce_scores = defaultdict(lambda: defaultdict(list))
    for idx, (q_idx, c_idx, ex_idx) in enumerate(pair_map):
        ce_scores[q_idx][c_idx].append(float(ce_scores_flat[idx]))

    # Now evaluate each query
    for q_idx in range(N):
        correct_label = facts[q_idx]["label"]
        template = facts[q_idx]["template_type"]
        alias_m = alias_sizes[q_idx]
        candidates = topK_indices[q_idx].tolist()
        candidates_20 = top20_indices[q_idx].tolist()

        # --- Oracle@K ---
        oracle_at_10 = 1 if correct_label in candidates else 0
        oracle_at_20 = 1 if correct_label in candidates_20 else 0

        # --- Baseline (prototypical L1, first-stage winner) ---
        baseline_pred = candidates[0]
        baseline_correct = 1 if baseline_pred == correct_label else 0

        # --- Token-Matching Re-Ranker ---
        query_tokens = query_token_sets[q_idx]
        token_class_scores_max = {}
        token_class_scores_mean = {}
        for c_idx in candidates:
            exemplar_scores = []
            for ex_idx in range(num_exemplars):
                score = token_overlap_score(query_tokens, facts[c_idx]["statements"][ex_idx])
                exemplar_scores.append(score)
            token_class_scores_max[c_idx] = max(exemplar_scores)
            token_class_scores_mean[c_idx] = np.mean(exemplar_scores)

        token_pred_max = max(candidates, key=lambda c: (token_class_scores_max[c], -dists[q_idx, c].item()))
        token_pred_mean = max(candidates, key=lambda c: (token_class_scores_mean[c], -dists[q_idx, c].item()))
        token_correct_max = 1 if token_pred_max == correct_label else 0
        token_correct_mean = 1 if token_pred_mean == correct_label else 0

        # --- Cross-Encoder Re-Ranker ---
        ce_class_scores_max = {}
        ce_class_scores_mean = {}
        for c_idx in candidates:
            scores_list = ce_scores[q_idx][c_idx]
            ce_class_scores_max[c_idx] = max(scores_list)
            ce_class_scores_mean[c_idx] = np.mean(scores_list)

        ce_pred_max = max(candidates, key=lambda c: ce_class_scores_max[c])
        ce_pred_mean = max(candidates, key=lambda c: ce_class_scores_mean[c])
        ce_correct_max = 1 if ce_pred_max == correct_label else 0
        ce_correct_mean = 1 if ce_pred_mean == correct_label else 0

        results.append({
            "query_idx": q_idx,
            "correct_label": correct_label,
            "template_type": template,
            "alias_m": alias_m,
            "oracle_at_10": oracle_at_10,
            "oracle_at_20": oracle_at_20,
            "baseline_correct": baseline_correct,
            "token_max_correct": token_correct_max,
            "token_mean_correct": token_correct_mean,
            "ce_max_correct": ce_correct_max,
            "ce_mean_correct": ce_correct_mean,
        })

    return results

# ==============================================================================
# 6. AGGREGATION & REPORTING
# ==============================================================================

def aggregate_results(results, N):
    """Aggregate per-query results into summary tables."""
    summary = {}

    # --- Overall ---
    summary["overall"] = {
        "N": N,
        "oracle_at_10": np.mean([r["oracle_at_10"] for r in results]),
        "oracle_at_20": np.mean([r["oracle_at_20"] for r in results]),
        "baseline": np.mean([r["baseline_correct"] for r in results]),
        "token_max": np.mean([r["token_max_correct"] for r in results]),
        "token_mean": np.mean([r["token_mean_correct"] for r in results]),
        "ce_max": np.mean([r["ce_max_correct"] for r in results]),
        "ce_mean": np.mean([r["ce_mean_correct"] for r in results]),
    }

    # --- Per-Template ---
    summary["per_template"] = {}
    for t in [0, 1, 2]:
        t_results = [r for r in results if r["template_type"] == t]
        if len(t_results) == 0:
            continue
        summary["per_template"][t] = {
            "count": len(t_results),
            "oracle_at_10": np.mean([r["oracle_at_10"] for r in t_results]),
            "oracle_at_20": np.mean([r["oracle_at_20"] for r in t_results]),
            "baseline": np.mean([r["baseline_correct"] for r in t_results]),
            "token_max": np.mean([r["token_max_correct"] for r in t_results]),
            "token_mean": np.mean([r["token_mean_correct"] for r in t_results]),
            "ce_max": np.mean([r["ce_max_correct"] for r in t_results]),
            "ce_mean": np.mean([r["ce_mean_correct"] for r in t_results]),
        }

    # --- Per Alias Cardinality ---
    summary["per_alias"] = {}
    alias_values = sorted(set(r["alias_m"] for r in results))
    for m in alias_values:
        m_results = [r for r in results if r["alias_m"] == m]
        if len(m_results) == 0:
            continue
        baseline_recall = np.mean([r["baseline_correct"] for r in m_results])
        ce_max_recall = np.mean([r["ce_max_correct"] for r in m_results])
        ce_mean_recall = np.mean([r["ce_mean_correct"] for r in m_results])
        token_max_recall = np.mean([r["token_max_correct"] for r in m_results])
        summary["per_alias"][m] = {
            "count": len(m_results),
            "bayes_ceiling": 1.0 / m,
            "oracle_at_10": np.mean([r["oracle_at_10"] for r in m_results]),
            "baseline": baseline_recall,
            "token_max": token_max_recall,
            "ce_max": ce_max_recall,
            "ce_mean": ce_mean_recall,
            "delta_ce_max": ce_max_recall - baseline_recall,
            "delta_ce_mean": ce_mean_recall - baseline_recall,
            "delta_token_max": token_max_recall - baseline_recall,
        }

    return summary

# ==============================================================================
# 7. CSV OUTPUT
# ==============================================================================

def write_csv(results, csv_path, N):
    """Write per-query results to CSV."""
    headers = [
        "N", "query_idx", "correct_label", "template_type", "alias_m",
        "oracle_at_10", "oracle_at_20",
        "baseline_correct", "token_max_correct", "token_mean_correct",
        "ce_max_correct", "ce_mean_correct"
    ]

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        for r in results:
            row = {"N": N}
            row.update(r)
            writer.writerow(row)

# ==============================================================================
# 8. PLOTTING
# ==============================================================================

def generate_plots(all_summaries, plot_path):
    """Generate diagnostic plots for the re-ranking evaluation."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Experiment 4: Relational Re-Ranker & Disambiguation Evaluation",
                 fontsize=16, fontweight='bold')

    # Use the largest N for detailed breakdowns
    largest_N = max(all_summaries.keys())
    summary = all_summaries[largest_N]

    # --- Plot 1: Overall Recall vs N ---
    ax = axes[0, 0]
    N_values = sorted(all_summaries.keys())
    methods = ["baseline", "token_max", "ce_max", "ce_mean"]
    labels = ["Baseline (L1)", "Token-Max", "CE-Max", "CE-Mean"]
    colors_list = ["#555555", "#E69F00", "#0072B2", "#D55E00"]

    for method, label, color in zip(methods, labels, colors_list):
        vals = [all_summaries[n]["overall"][method] * 100 for n in N_values]
        ax.plot(N_values, vals, 'o-', label=label, color=color, linewidth=2, markersize=6)

    # Oracle line
    oracle_vals = [all_summaries[n]["overall"]["oracle_at_10"] * 100 for n in N_values]
    ax.plot(N_values, oracle_vals, 's--', label="Oracle@10", color="#009E73",
            linewidth=2, markersize=6)

    ax.set_xlabel("Number of Classes (N)")
    ax.set_ylabel("Top-1 Recall (%)")
    ax.set_title("Overall Recall vs N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Per-Template Recall (at largest N) ---
    ax = axes[0, 1]
    template_names = {0: "T0: Access Codes", 1: "T1: Folders/Projects", 2: "T2: Meetings"}
    x_positions = np.arange(3)
    bar_width = 0.15

    for i, (method, label, color) in enumerate(zip(methods, labels, colors_list)):
        vals = []
        for t in [0, 1, 2]:
            if t in summary["per_template"]:
                vals.append(summary["per_template"][t][method] * 100)
            else:
                vals.append(0)
        ax.bar(x_positions + i * bar_width, vals, bar_width, label=label, color=color)

    # Oracle bars
    oracle_vals_t = []
    for t in [0, 1, 2]:
        if t in summary["per_template"]:
            oracle_vals_t.append(summary["per_template"][t]["oracle_at_10"] * 100)
        else:
            oracle_vals_t.append(0)
    ax.bar(x_positions + 4 * bar_width, oracle_vals_t, bar_width, label="Oracle@10",
           color="#009E73")

    ax.set_xticks(x_positions + 2 * bar_width)
    ax.set_xticklabels([template_names[t] for t in [0, 1, 2]], fontsize=8)
    ax.set_ylabel("Top-1 Recall (%)")
    ax.set_title(f"Per-Template Recall (N={largest_N})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Plot 3: Recall vs Alias Cardinality ---
    ax = axes[0, 2]
    alias_ms = sorted(summary["per_alias"].keys())
    if alias_ms:
        baseline_by_m = [summary["per_alias"][m]["baseline"] * 100 for m in alias_ms]
        ce_max_by_m = [summary["per_alias"][m]["ce_max"] * 100 for m in alias_ms]
        bayes_by_m = [summary["per_alias"][m]["bayes_ceiling"] * 100 for m in alias_ms]
        oracle_by_m = [summary["per_alias"][m]["oracle_at_10"] * 100 for m in alias_ms]

        ax.plot(alias_ms, baseline_by_m, 'o-', label="Baseline", color="#555555",
                linewidth=2, markersize=6)
        ax.plot(alias_ms, ce_max_by_m, 's-', label="CE-Max", color="#0072B2",
                linewidth=2, markersize=6)
        ax.plot(alias_ms, bayes_by_m, 'x--', label="Bayes Ceiling (1/M)", color="#CC79A7",
                linewidth=2, markersize=8)
        ax.plot(alias_ms, oracle_by_m, 'd--', label="Oracle@10", color="#009E73",
                linewidth=2, markersize=6)

        ax.set_xlabel("Alias Cardinality (M)")
        ax.set_ylabel("Top-1 Recall (%)")
        ax.set_title(f"Recall vs Alias Cardinality (N={largest_N})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Plot 4: Reranker Gain (Δ) vs Alias Cardinality ---
    ax = axes[1, 0]
    if alias_ms:
        delta_ce_max = [summary["per_alias"][m]["delta_ce_max"] * 100 for m in alias_ms]
        delta_token = [summary["per_alias"][m]["delta_token_max"] * 100 for m in alias_ms]

        ax.bar(np.arange(len(alias_ms)) - 0.15, delta_ce_max, 0.3, label="Δ CE-Max",
               color="#0072B2")
        ax.bar(np.arange(len(alias_ms)) + 0.15, delta_token, 0.3, label="Δ Token-Max",
               color="#E69F00")

        ax.set_xticks(np.arange(len(alias_ms)))
        ax.set_xticklabels([str(m) for m in alias_ms])
        ax.set_xlabel("Alias Cardinality (M)")
        ax.set_ylabel("Recall Gain Δ (pp)")
        ax.set_title(f"Reranker Gain vs Alias Cardinality (N={largest_N})")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    # --- Plot 5: Oracle@K per Template ---
    ax = axes[1, 1]
    if summary["per_template"]:
        templates = [0, 1, 2]
        oracle_10_vals = [summary["per_template"].get(t, {}).get("oracle_at_10", 0) * 100
                          for t in templates]
        oracle_20_vals = [summary["per_template"].get(t, {}).get("oracle_at_20", 0) * 100
                          for t in templates]

        x_pos = np.arange(3)
        ax.bar(x_pos - 0.15, oracle_10_vals, 0.3, label="Oracle@10", color="#009E73")
        ax.bar(x_pos + 0.15, oracle_20_vals, 0.3, label="Oracle@20", color="#56B4E9")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([template_names[t] for t in templates], fontsize=8)
        ax.set_ylabel("Coverage (%)")
        ax.set_title(f"Oracle@K by Template (N={largest_N})")
        ax.legend(fontsize=8)
        ax.set_ylim(90, 101)
        ax.grid(True, alpha=0.3, axis='y')

    # --- Plot 6: Alias Set Size Distribution ---
    ax = axes[1, 2]
    if alias_ms:
        counts = [summary["per_alias"][m]["count"] for m in alias_ms]
        ax.bar(range(len(alias_ms)), counts, color="#56B4E9")
        ax.set_xticks(range(len(alias_ms)))
        ax.set_xticklabels([str(m) for m in alias_ms])
        ax.set_xlabel("Alias Cardinality (M)")
        ax.set_ylabel("Number of Queries")
        ax.set_title(f"Alias Set Size Distribution (N={largest_N})")
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Plots saved to {plot_path}")

# ==============================================================================
# 9. MARKDOWN REPORT
# ==============================================================================

def generate_report(all_summaries, md_path):
    """Generate a markdown summary report."""
    largest_N = max(all_summaries.keys())
    summary = all_summaries[largest_N]

    lines = []
    lines.append("# Experiment 4: Relational Re-Ranker & Disambiguation Evaluation\n")
    lines.append(f"**Evaluated at N = {sorted(all_summaries.keys())}**\n")
    lines.append("---\n")

    # Overall table
    lines.append("## 1. Overall Results\n")
    lines.append("| N | Oracle@10 | Oracle@20 | Baseline | Token-Max | CE-Max | CE-Mean |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for n in sorted(all_summaries.keys()):
        s = all_summaries[n]["overall"]
        lines.append(f"| {n} | {s['oracle_at_10']*100:.1f}% | {s['oracle_at_20']*100:.1f}% | "
                      f"{s['baseline']*100:.1f}% | {s['token_max']*100:.1f}% | "
                      f"{s['ce_max']*100:.1f}% | {s['ce_mean']*100:.1f}% |")
    lines.append("")

    # Per-template table
    lines.append("## 2. Per-Template Results (N={})".format(largest_N))
    lines.append("")
    template_names = {0: "T0: Access Codes", 1: "T1: Folders/Projects", 2: "T2: Meetings"}
    lines.append("| Template | Count | Oracle@10 | Baseline | Token-Max | CE-Max | CE-Mean |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for t in [0, 1, 2]:
        if t in summary["per_template"]:
            s = summary["per_template"][t]
            lines.append(f"| {template_names[t]} | {s['count']} | {s['oracle_at_10']*100:.1f}% | "
                          f"{s['baseline']*100:.1f}% | {s['token_max']*100:.1f}% | "
                          f"{s['ce_max']*100:.1f}% | {s['ce_mean']*100:.1f}% |")
    lines.append("")

    # Alias cardinality table
    lines.append("## 3. Recall vs Alias Cardinality (N={})".format(largest_N))
    lines.append("")
    lines.append("| M | Count | Bayes Ceiling | Oracle@10 | Baseline | CE-Max | Δ CE-Max | Δ Token-Max |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in sorted(summary["per_alias"].keys()):
        s = summary["per_alias"][m]
        lines.append(f"| {m} | {s['count']} | {s['bayes_ceiling']*100:.1f}% | "
                      f"{s['oracle_at_10']*100:.1f}% | {s['baseline']*100:.1f}% | "
                      f"{s['ce_max']*100:.1f}% | {s['delta_ce_max']*100:+.1f}pp | "
                      f"{s['delta_token_max']*100:+.1f}pp |")
    lines.append("")

    # Error decomposition
    lines.append("## 4. Error Decomposition (N={})".format(largest_N))
    lines.append("")
    oracle = summary["overall"]["oracle_at_10"]
    ce_max_recall = summary["overall"]["ce_max"]
    lines.append(f"- **Retrieval Error** = 100% - Oracle@10 = {(1.0 - oracle)*100:.1f}%")
    lines.append(f"- **Selection Error** = Oracle@10 - CE-Max Recall = {(oracle - ce_max_recall)*100:.1f}%")
    lines.append(f"- **Total Error** = 100% - CE-Max Recall = {(1.0 - ce_max_recall)*100:.1f}%")
    lines.append("")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  [+] Report saved to {md_path}")

# ==============================================================================
# 10. MAIN
# ==============================================================================

def main():
    set_seed(42)

    print("====================================================")
    print("  Experiment 4: Relational Re-Ranker Evaluation")
    print("  (Kaggle Self-Contained Script)")
    print("====================================================")

    # Save directly to Kaggle's working directory
    results_dir = "."
    csv_path = os.path.join(results_dir, "relational_reranker.csv")
    plot_path = os.path.join(results_dir, "relational_reranker.png")
    md_path = os.path.join(results_dir, "relational_reranker.md")

    K = 10

    # Clear CSV if exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Load models
    print("[*] Loading bi-encoder: all-MiniLM-L6-v2...")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    bi_encoder.to(device)

    print("[*] Loading cross-encoder: ms-marco-MiniLM-L-6-v2...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    N_values = [100, 200, 400, 800]

    all_summaries = {}

    for N in N_values:
        print(f"\n{'='*60}")
        print(f"  Running N = {N}")
        print(f"{'='*60}")

        t0 = time.time()
        facts = generate_facts(N)

        # Compute and display alias statistics
        alias_sizes, query_to_labels = compute_alias_sets(facts)
        alias_counter = Counter(alias_sizes)
        print(f"  [*] Alias cardinality distribution:")
        for m in sorted(alias_counter.keys()):
            print(f"      M={m}: {alias_counter[m]} queries")

        # Embed statements and queries
        print(f"  [*] Encoding {N * 10} statements and {N} queries...")
        all_statement_texts = [s for f in facts for s in f["statements"]]
        all_query_texts = [f["query"] for f in facts]

        with torch.no_grad():
            statements_flat = bi_encoder.encode(all_statement_texts, batch_size=256,
                                                 convert_to_tensor=True, device=device)
            statements_emb = statements_flat.view(N, 10, -1)
            queries_emb = bi_encoder.encode(all_query_texts, batch_size=256,
                                             convert_to_tensor=True, device=device)

        # Move to CPU for distance computation (save GPU memory)
        statements_emb_cpu = statements_emb.cpu()
        queries_emb_cpu = queries_emb.cpu()

        # Run re-ranking evaluation
        print(f"  [*] Running re-ranking evaluation (K={K})...")
        results = evaluate_rerankers(facts, statements_emb_cpu, queries_emb_cpu,
                                      cross_encoder, K=K, eval_device='cpu')

        # Aggregate
        summary = aggregate_results(results, N)
        all_summaries[N] = summary

        # Write CSV
        write_csv(results, csv_path, N)

        # Print summary
        s = summary["overall"]
        elapsed = time.time() - t0
        print(f"\n  --- Results for N={N} (took {elapsed:.1f}s) ---")
        print(f"  Oracle@10:     {s['oracle_at_10']*100:.1f}%")
        print(f"  Oracle@20:     {s['oracle_at_20']*100:.1f}%")
        print(f"  Baseline:      {s['baseline']*100:.1f}%")
        print(f"  Token-Max:     {s['token_max']*100:.1f}%")
        print(f"  Token-Mean:    {s['token_mean']*100:.1f}%")
        print(f"  CE-Max:        {s['ce_max']*100:.1f}%")
        print(f"  CE-Mean:       {s['ce_mean']*100:.1f}%")

        print(f"\n  Per-Template:")
        template_names = {0: "T0:AccessCodes", 1: "T1:Folders", 2: "T2:Meetings"}
        for t in [0, 1, 2]:
            if t in summary["per_template"]:
                ts = summary["per_template"][t]
                print(f"    {template_names[t]:16s} | Oracle@10={ts['oracle_at_10']*100:.1f}% "
                      f"| Base={ts['baseline']*100:.1f}% | CE-Max={ts['ce_max']*100:.1f}%")

        print(f"\n  Alias Cardinality:")
        for m in sorted(summary["per_alias"].keys()):
            ms = summary["per_alias"][m]
            print(f"    M={m:2d} ({ms['count']:3d} queries) | "
                  f"Bayes={ms['bayes_ceiling']*100:.1f}% | Base={ms['baseline']*100:.1f}% | "
                  f"CE-Max={ms['ce_max']*100:.1f}% | Δ={ms['delta_ce_max']*100:+.1f}pp")

    # Generate plots and report
    print(f"\n[*] Generating plots...")
    generate_plots(all_summaries, plot_path)

    print(f"[*] Generating markdown report...")
    generate_report(all_summaries, md_path)

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  CSV:    {csv_path}")
    print(f"  Plots:  {plot_path}")
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    main()
