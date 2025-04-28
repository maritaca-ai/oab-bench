"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures

import shortuuid
import tqdm

from common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
)
from conversation import get_conv_template as get_conversation_template


def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str,
    api_base: str = None, api_key: str = None,
):
    assert (
        args.force_temperature is not None and "required_temperature" in question.keys()
    ) == False
    if args.force_temperature is not None:
        temperature = args.force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    for i in range(num_choices):
        conv = get_conversation_template("chatgpt")

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            api_dict = {}
            if api_key is not None:
                api_dict["api_key"] = api_key
            if api_base is not None:
                api_dict["api_base"] = api_base
                
            output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-base", type=str, default=None)
    parser.add_argument("--openai-key-env", type=str, default=None)
    args = parser.parse_args()

    api_key = None
    if args.openai_key_env is not None:
        api_key = os.environ[args.openai_key_env]

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    # Exams: if the question has a statement, add it as a prefix in the first turn
    for q in questions:
        if 'statement' in q:
            q['turns'][0] = q['statement'] + '\n' + q['turns'][0]

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model.split('/')[-1]}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
                api_base=args.openai_api_base,
                api_key=api_key,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
