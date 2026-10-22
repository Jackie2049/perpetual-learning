import datasets, re, json, os

parquet_train = "/home/zxw/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet"
parquet_test = "/home/zxw/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"
ds = datasets.Dataset.from_parquet(parquet_train)
data_source = "openai/gsm8k"
instruction_following = "Let's think step by step and output the final answer after `####`."

def extract_solution(s):
    m = re.search("#### (\\-?[0-9\\.\\,]+)", s)
    assert m is not None
    return m.group(0).split("#### ")[1].replace(",", "")

def process_fn(example, idx):
    question_raw = example.pop("question")
    answer_raw = example.pop("answer")
    solution = extract_solution(answer_raw)
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": "You are a math expert. Solve the question step by step, use the calc_gsm8k_reward tool, and put your final answer in `#### <answer>`."},
            {"role": "user", "content": question_raw + " " + instruction_following},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": "train", "index": idx, "answer": answer_raw, "question": question_raw,
            "need_tools_kwargs": True,
            "tools_kwargs": {"calc_gsm8k_reward": {"create_kwargs": {"ground_truth": solution}}},
        },
    }

# smoke: keep first 16 train + 4 test rows
train = ds.map(process_fn, with_indices=True).select(range(16))
test = datasets.Dataset.from_parquet(parquet_test).map(process_fn, with_indices=True).select(range(4))
os.makedirs("/tmp/gsm8k_agent", exist_ok=True)
train.to_parquet("/tmp/gsm8k_agent/train.parquet")
test.to_parquet("/tmp/gsm8k_agent/test.parquet")
print("train rows:", len(train), "test rows:", len(test))
print("sample keys:", list(train[0].keys()))
