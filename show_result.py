"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import glob
import json
import os
from collections import defaultdict
import pandas as pd
import warnings


def deduplicate_judgments(df):
    """
    Remove duplicate judgments keeping only the last occurrence.
    Duplicates are identified by question_id, model, and turn.
    
    Args:
        df: DataFrame with model judgments
        
    Returns:
        DataFrame with duplicates removed
    """
    # Check for duplicates based on key columns
    key_cols = ['question_id', 'model', 'turn']
    
    # Find duplicates
    duplicates_mask = df.duplicated(subset=key_cols, keep=False)
    
    if duplicates_mask.any():
        duplicate_count = duplicates_mask.sum()
        duplicate_groups = df.duplicated(subset=key_cols, keep='last').sum()
        
        warnings.warn(
            f"Found {duplicate_count} duplicate judgments ({duplicate_groups} duplicated entries). "
            f"Keeping only the last occurrence of each (question_id, model, turn) combination.",
            UserWarning
        )
        
        # Print details about duplicates for debugging
        print(f"\n=== Duplicate Detection ===")
        print(f"Total duplicate entries found: {duplicate_count}")
        print(f"Unique combinations with duplicates: {duplicate_groups}")
        
        # Show some examples of duplicates
        duplicate_examples = df[duplicates_mask].groupby(key_cols).size()
        print(f"\nExamples of duplicate combinations:")
        for (qid, model, turn), count in duplicate_examples.head(10).items():
            print(f"  {qid}, {model}, turn {turn}: {count} occurrences")
        if len(duplicate_examples) > 10:
            print(f"  ... and {len(duplicate_examples) - 10} more")
    
    # Keep only the last occurrence of each duplicate
    df_deduped = df.drop_duplicates(subset=key_cols, keep='last')
    
    return df_deduped


def new_usage_totals():
    return {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}


def normalize_usage(usage):
    if usage is None:
        return None
    if isinstance(usage, float) and pd.isna(usage):
        return None
    if isinstance(usage, dict):
        return usage
    return None


def extract_token_counts(usage):
    """Return (prompt_tokens, completion_tokens, total_tokens) from a usage dict."""
    if not usage:
        return None, None, None

    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")

    # allow nested details to provide fallbacks
    if prompt is None and isinstance(usage.get("prompt_tokens_details"), dict):
        prompt_details = usage["prompt_tokens_details"]
        prompt = sum(v for v in prompt_details.values() if isinstance(v, (int, float)))

    if completion is None and isinstance(usage.get("completion_tokens_details"), dict):
        comp_details = usage["completion_tokens_details"]
        completion = sum(v for v in comp_details.values() if isinstance(v, (int, float)))

    return prompt, completion, total


def add_usage(acc, usage, weight=1.0):
    usage = normalize_usage(usage)
    if not usage:
        return

    prompt, completion, total = extract_token_counts(usage)

    if prompt is not None:
        acc["prompt_tokens"] += prompt * weight
    if completion is not None:
        acc["completion_tokens"] += completion * weight
    if total is not None:
        acc["total_tokens"] += total * weight
    elif prompt is not None or completion is not None:
        acc["total_tokens"] += ((prompt or 0) + (completion or 0)) * weight


def load_generation_usage(bench_name, models):
    usage_by_model = defaultdict(new_usage_totals)
    answer_dir = f"data/{bench_name}/model_answer"
    for path in glob.glob(os.path.join(answer_dir, "*.jsonl")):
        model_name = os.path.splitext(os.path.basename(path))[0]
        if models and model_name not in models:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    for choice in obj.get("choices", []):
                        for turn in choice.get("turns", []):
                            usage = turn.get("usage") if isinstance(turn, dict) else None
                            add_usage(usage_by_model[model_name], usage)
        except FileNotFoundError:
            continue
    return usage_by_model


def aggregate_judge_usage_single(df, models):
    usage_by_model = defaultdict(new_usage_totals)
    for _, row in df.iterrows():
        model = row.get("model")
        if model not in models:
            continue
        add_usage(usage_by_model[model], row.get("judge_usage"))
    return usage_by_model


def aggregate_judge_usage_pairwise(df, models, baseline_model=None):
    usage_by_model = defaultdict(new_usage_totals)
    for _, row in df.iterrows():
        model_1 = row.get("model_1")
        model_2 = row.get("model_2")
        g1_usage = row.get("g1_usage")
        g2_usage = row.get("g2_usage")

        participants = [model_1, model_2]
        if baseline_model is not None and baseline_model in participants:
            targets = [m for m in participants if m != baseline_model and m in models]
            weight = 1.0
        else:
            targets = [m for m in participants if m in models]
            weight = 0.5

        for target in targets:
            add_usage(usage_by_model[target], g1_usage, weight)
            add_usage(usage_by_model[target], g2_usage, weight)
    return usage_by_model


def build_usage_table(models, gen_usage, judge_usage):
    rows = []
    for model in sorted(models):
        gen = gen_usage.get(model, new_usage_totals())
        judge = judge_usage.get(model, new_usage_totals())

        total_prompt = gen["prompt_tokens"] + judge["prompt_tokens"]
        total_completion = gen["completion_tokens"] + judge["completion_tokens"]
        total_tokens = gen["total_tokens"] + judge["total_tokens"]
        if total_tokens == 0:
            total_tokens = total_prompt + total_completion

        row = {
            "model": model,
            "gen_prompt_tokens": gen["prompt_tokens"],
            "gen_completion_tokens": gen["completion_tokens"],
            "judge_prompt_tokens": judge["prompt_tokens"],
            "judge_completion_tokens": judge["completion_tokens"],
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
        }

        rows.append(row)

    if not rows:
        return None

    usage_df = pd.DataFrame(rows).set_index("model")
    numeric_cols = [
        "gen_prompt_tokens",
        "gen_completion_tokens",
        "judge_prompt_tokens",
        "judge_completion_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
    ]
    for col in numeric_cols:
        if col in usage_df.columns:
            usage_df[col] = usage_df[col].astype(float)

    return usage_df


def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    
    # Remove duplicates keeping only the last occurrence
    df_all = deduplicate_judgments(df_all)
    df_usage_source = df_all.copy()
    all_exams = [1]

    if args.bench_name == 'oab_bench':
        # for each question, sum the scores of all subquestions
        df_all = df_all.groupby(['question_id', 'model'], as_index=False)['score'].sum()
        
        # extract the exam identifier from the question_id
        df_all['exam'] = df_all['question_id'].apply(lambda x: '_'.join(x.split('_')[:-2]))
        
        # create dictionary to aggregate scores
        agg_dict = {
            'score': 'sum',  # sum total (overall)
        }
        
        # add columns for each individual question
        question_ids = df_all['question_id'].unique()
        for qid in question_ids:
            df_all[f'q{qid}_score'] = df_all.apply(
                lambda row: row['score'] if row['question_id'] == qid else 0,
                axis=1
            )
            agg_dict[f'q{qid}_score'] = 'sum'
        
        # group by model to have both overall and individual questions
        df_all = df_all.groupby(['model'], as_index=False).agg(agg_dict)
        df_all["turn"] = 1
        
        # calculate and display means by model, separated by exam
        print("\n=== Scores by Exam ===")
        
        # identify all unique exams
        all_exams = sorted(set('_'.join(qid.split('_')[:-2]) for qid in question_ids))
        
        # Initialize a dictionary to count approved exams per model
        approved_exams = defaultdict(int)
        
        # Create a DataFrame for exam scores
        exam_scores = pd.DataFrame(index=all_exams)
        
        for exam in all_exams:
            print(f"\n--- {exam} ---")
            exam_questions = [qid for qid in question_ids if exam in qid]
            
            means_by_model = df_all.groupby('model').agg({
                **{f'q{qid}_score': 'first' for qid in exam_questions}
            })
            
            # calculate the total only for the questions of this exam
            means_by_model['total'] = means_by_model[[f'q{qid}_score' for qid in exam_questions]].sum(axis=1)
            
            # rename columns for better understanding
            means_by_model = means_by_model.rename(columns={
                **{f'q{qid}_score': f'questao_{qid.split("_")[-1]}' for qid in exam_questions}
            })
            print(means_by_model.round(4).sort_values(by='total', ascending=False))
            
            # Add total scores to exam_scores DataFrame
            # exam_scores[means_by_model.index] = means_by_model['total']
            # Fix: Use proper column assignment
            for model in means_by_model.index:
                exam_scores.loc[exam, model] = means_by_model.loc[model, 'total']
            
            # Count approved exams (score >= 6.0) for each model
            for model, row in means_by_model.iterrows():
                if row['total'] >= 6.0:
                    approved_exams[model] += 1
        
        print("\n=== Number of Approved Exams per Model (score >= 6.0) ===")
        for model, count in sorted(approved_exams.items(), key=lambda x: x[1], reverse=True):
            print(f"{model}: {count}/{len(all_exams)} exams")
    
    df_filtered = df_all[df_all["score"] != -1]
    if args.model_list is not None:
        df_filtered = df_filtered[df_filtered["model"].isin(args.model_list)]

    df = df_filtered[["model", "score", "turn"]]

    exam_divisor = len(all_exams) if all_exams else 1
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean() / exam_divisor
    print(df_1.sort_values(by="score", ascending=False))

    models_for_usage = set(df_1.index.get_level_values(0))
    usage_table = None
    if models_for_usage and (args.show_usage or args.wandb_project is not None):
        usage_filtered = df_usage_source[df_usage_source["score"] != -1]
        if args.model_list is not None:
            usage_filtered = usage_filtered[usage_filtered["model"].isin(args.model_list)]
        judge_usage = aggregate_judge_usage_single(usage_filtered, models_for_usage)
        gen_usage = load_generation_usage(args.bench_name, models_for_usage)
        usage_table = build_usage_table(models_for_usage, gen_usage, judge_usage)

        if args.show_usage and usage_table is not None:
            print("\n=== Token usage per model (generation + judge) ===")
            print(usage_table.round(2))

    if args.wandb_project is not None:
        assert args.wandb_experiment_name is not None, "wandb_experiment_name must be specified when args.wandb_project is set"
        assert args.wandb_entity is not None, "wandb_entity must be specified when args.wandb_project is set"
        assert args.wandb_model_id is not None, "wandb_model_id must be specified when args.wandb_project is set"

        # Log the results to wandb only for the model specified by wandb_model_id
        model_scores = df_1[df_1.index.get_level_values(0) == args.wandb_model_id]
        assert not model_scores.empty, f"No scores found for model {args.wandb_model_id}"

        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_experiment_name,
            resume="allow",
            config={
                "bench_name": args.bench_name,
                "judge_model": args.judge_model,
                "wandb_model_id": args.wandb_model_id,
            },
        )

        log_data = {
            f"score_{args.wandb_model_id}": model_scores['score'].values[0],
        }

        if usage_table is not None and args.wandb_model_id in usage_table.index:
            usage_row = usage_table.loc[args.wandb_model_id]
            log_data.update({
                f"tokens_prompt_{args.wandb_model_id}": float(usage_row["total_prompt_tokens"]),
                f"tokens_completion_{args.wandb_model_id}": float(usage_row["total_completion_tokens"]),
                f"tokens_total_{args.wandb_model_id}": float(usage_row["total_tokens"]),
                f"tokens_generation_prompt_{args.wandb_model_id}": float(usage_row["gen_prompt_tokens"]),
                f"tokens_generation_completion_{args.wandb_model_id}": float(usage_row["gen_completion_tokens"]),
                f"tokens_judge_prompt_{args.wandb_model_id}": float(usage_row["judge_prompt_tokens"]),
                f"tokens_judge_completion_{args.wandb_model_id}": float(usage_row["judge_completion_tokens"]),
            })

        wandb.log(log_data)
        
        print(f"Results logged to wandb: {args.wandb_project}/{args.wandb_experiment_name}")

def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    
    # Remove duplicates keeping only the last occurrence
    df_all = deduplicate_judgments(df_all)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))

    list_res = []
    filtered_rows = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        filtered_rows.append(row)
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))

    models_for_usage = set(df.index)
    usage_table = None
    if models_for_usage and (args.show_usage or args.wandb_project is not None):
        usage_source = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame(columns=df_all.columns)
        judge_usage = aggregate_judge_usage_pairwise(
            usage_source,
            models_for_usage,
            baseline_model=args.baseline_model,
        )
        gen_usage = load_generation_usage(args.bench_name, models_for_usage)
        usage_table = build_usage_table(models_for_usage, gen_usage, judge_usage)

        if args.show_usage and usage_table is not None:
            print("\n=== Token usage per model (generation + judge) ===")
            print(usage_table.round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="oab_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="o1-2024-12-17")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Show token usage aggregated per model (generation + judge).",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-experiment-name", type=str, default=None)
    parser.add_argument(
        "--wandb-model-id", type=str, default=None, 
        help="Model ID for wandb logging. This script computes scores for all judged models."
             "However, we only log to wandb one model, which is specified in this argument.")

    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
