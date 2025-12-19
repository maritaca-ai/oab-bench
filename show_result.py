"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from collections import defaultdict
import pandas as pd
import warnings
import flatten_dict

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
        print("\n=== Duplicate Detection ===")
        print(f"Total duplicate entries found: {duplicate_count}")
        print(f"Unique combinations with duplicates: {duplicate_groups}")
        
        # Show some examples of duplicates
        duplicate_examples = df[duplicates_mask].groupby(key_cols).size()
        print("\nExamples of duplicate combinations:")
        for (qid, model, turn), count in duplicate_examples.head(10).items():
            print(f"  {qid}, {model}, turn {turn}: {count} occurrences")
        if len(duplicate_examples) > 10:
            print(f"  ... and {len(duplicate_examples) - 10} more")
    
    # Keep only the last occurrence of each duplicate
    df_deduped = df.drop_duplicates(subset=key_cols, keep='last')
    
    return df_deduped

def add_level_to_flat_dict(d, name):
    return {f"{name}/{k}": v for k, v in d.items()}


def normalize_flatten_usage(d, parent_key=None):
    """Flatten a usage dict and apply defaults so aggregation/join math stays safe."""
    out = flatten_dict.flatten({k: v for k, v in d.items() if v}, "path")
    if parent_key is not None:
        out = add_level_to_flat_dict(out, parent_key)
    return out


def calculate_token_usage_single(df_all, model_list, bench_name: str):
    """Aggregate judge and answer token usage per model for single-mode reports."""
    # Select only necessary columns and drop null data (can only account for valid token usage)

    df_judge_usage = (
        df_all[["model", "judge", "judgment_id", "answer_id", "judge_usage"]]
        .dropna()
        .copy()
    )
    df_judge_usage["judge"] = df_judge_usage["judge"].str[0]
    if model_list is not None:
        df_judge_usage = df_judge_usage[df_judge_usage["model"].isin(model_list)]
    df_judge_usage = (
        df_judge_usage.set_index(
            ["model", "judge", "answer_id", "judgment_id"], verify_integrity=True
        )[
            # Flatten judge_usage so later joins/sums work on columns
            "judge_usage"
        ]
        .apply(lambda x: normalize_flatten_usage(x, "judge"))
        .to_frame()
    )
    # Load only the judged models' answers and flatten their usage
    models_to_load = df_judge_usage.index.get_level_values("model").unique()
    df_answer_usage = (
        # Concatenate usage data from all models
        pd.concat(
            [
                pd.read_json(
                    f"data/{bench_name}/model_answer/{model}.jsonl", lines=True
                ).assign(model=model)
                for model in models_to_load
            ]
        )
        # Set index for joining
        .set_index(["model", "answer_id"], verify_integrity=True)["choices"]
        # Take the first choice and flatten usage for every turn
        .map(
            lambda x: [
                normalize_flatten_usage(turn["usage"], "answer")
                for turn in x[0]["turns"]
            ]
        )
        # Explode to account for each turn and convert to a dataframe for joining
        .explode()
        .to_frame("answer_usage")
    )
    answer_ids = df_judge_usage.index.get_level_values("answer_id").unique()
    df_answer_usage = (
        df_answer_usage[
            df_answer_usage.index.get_level_values("answer_id").isin(answer_ids)
        ]
        .reset_index("answer_id", drop=True)["answer_usage"]
        # Convert a Series of dicts to a dataframe and total tokens per model
        .apply(pd.Series)
        .groupby("model")
        .sum()
        .reset_index()
    )
    # Sum judge and answer usage per model to compare total cost
    df_judge_usage = (
        df_judge_usage["judge_usage"]
        .apply(pd.Series)
        .groupby(["model", "judge"])
        .sum()
        .reset_index()
    )
    df_usage = df_judge_usage.merge(
        df_answer_usage,
        on="model",
        how="left",
        validate="m:1",
    )
    return df_usage


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
    df_token_usage = calculate_token_usage_single(
        df_all, args.model_list, args.bench_name
    )

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
    

    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean() / len(all_exams)
    print(df_1.sort_values(by="score", ascending=False))
    
    if not df_token_usage.empty:
        print("\n=== Token usage per model (generation + judge) ===")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", 0,
            "display.max_colwidth", None,
            "display.width", 0,
            "display.expand_frame_repr", True,
            "display.float_format", lambda x: f"{x:_.6f}".rstrip("0").rstrip("."),
        ):
            main_token_usage_cols = [
                "judge/prompt_tokens",
                "judge/completion_tokens",
                "judge/completion_tokens_details/reasoning_tokens",
                "judge/prompt_tokens_details/cached_tokens",
                "answer/prompt_tokens",
                "answer/completion_tokens",
                "answer/completion_tokens_details/reasoning_tokens",
                "answer/prompt_tokens_details/cached_tokens",
            ]
            df_usage_pretty = df_token_usage.set_index("model", verify_integrity=True)
            df_usage_pretty = (
                df_usage_pretty.loc[
                    :, df_usage_pretty.columns.isin(main_token_usage_cols)
                ]
                .astype(float)
                .copy()
            )
            df_usage_pretty.columns = df_usage_pretty.columns.str.split(
                "/", n=1, expand=True
            )
            df_usage_pretty = df_usage_pretty.stack().add_suffix("_tokens")
            print(df_usage_pretty)
            print("Note: showing only columns with positive values; see W&B for the full table.")
    

    if args.wandb_project is not None:
        assert args.wandb_experiment_name is not None, "wandb_experiment_name must be specified when args.wandb_project is set"
        assert args.wandb_entity is not None, "wandb_entity must be specified when args.wandb_project is set"
        assert args.wandb_model_id is not None, "wandb_model_id must be specified when args.wandb_project is set"

        # Log the results to wandb only for the model specified by wandb_model_id
        model_scores = df_1[df_1.index.get_level_values(0) == args.wandb_model_id]
        assert not model_scores.empty, f"No scores found for model {args.wandb_model_id}"

        token_usage_for_model = {}
        if not df_token_usage.empty:
            token_usage_for_model = (
                df_token_usage.set_index(["model", "judge"], verify_integrity=True)
                .to_dict("index")
                .get((args.wandb_model_id, args.judge_model), {})
            )
        if token_usage_for_model:
            token_usage_for_model = add_level_to_flat_dict(token_usage_for_model, "token_usage")
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
        wandb_table = wandb.Table(
            columns=["run_id", "model", "judge_model", "score", *token_usage_for_model.keys()],
            rows=[
                [
                    args.wandb_experiment_name,
                    args.wandb_model_id,
                    args.judge_model,
                    model_scores["score"].values[0],
                    *token_usage_for_model.values(),
                ],
            ],
            log_mode="MUTABLE",
        )

        wandb.log({
            f"score_{args.wandb_model_id}": model_scores['score'].values[0],
            **token_usage_for_model,
            "results_table": wandb_table,
        })
        
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
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
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
