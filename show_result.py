"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from collections import defaultdict
import pandas as pd


def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)

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

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
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
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)
