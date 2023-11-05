# ToolLLM + ToolDec

## Repository Structure

`toolbench`: A modified version of ToolLLM to use with ToolDec. 
- Note that the modification is based on a [8/20/23 Snapshot](https://github.com/OpenBMB/ToolBench/tree/745ea4c3d670e83ef75697896016fce8ef9c6ea0). Please refer to the original repository for more up-to-date versions.
- Code unrelated to ToolDec experiments (such as training) are removed.

`data/query`: ToolEval test sets used in experiments. Query files contain testcases extracted from the original tests. 

`data/output`: Output of the experiments. `nofuzzy` refers to the extrapolation study without fuzzy matching.

`data/toolenv`: Python wrappers of the APIs.

## Preparation

```sh
pip install -r requirements.txt
```

```
unzip data.zip
```

Please refer to [this guide](https://github.com/OpenBMB/ToolBench/tree/master#inference-with-our-rapidapi-server) to obtain a ToolBench key.

```sh
export TOOLBENCH_KEY="key"
```

```sh
export PYTHONPATH=./
```

## Inference Pipelines

```sh
export TEST_SET="i2_cat" # for I2-Category
```

or

```sh
export TEST_SET="i3_inst" # for I3-Instruction
```

### ChatGPT

```sh
export OPENAI_KEY="key"

python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model chatgpt_function \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file "data/query/${TEST_SET}.json" \
    --output_answer_file "data/output/chatgpt_${TEST_SET}_cot_18" \
    --toolbench_key $TOOLBENCH_KEY
```

### ToolLLM

```sh
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path ToolBench/ToolLLaMA-7b \
    --method CoT@1 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --input_query_file "data/query/${TEST_SET}.json" \
    --output_answer_file "data/output/toolllm_${TEST_SET}_cot_18" \
    --toolbench_key $TOOLBENCH_KEY 
```

To test without Fuzzy Matching, change line 335 in [rapidapi.py](./toolbench/inference/Downstream_tasks/rapidapi.py) 
from 
```python
if function["name"].endswith(action_name):
```
to 
```python
if function["name"] == action_name:
```

### ToolLLM + ToolDec

```sh
python3 toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path ToolBench/ToolLLaMA-7b \
    --method CoT@1 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --input_query_file "data/query/${TEST_SET}.json" \
    --output_answer_file "data/output/toolllm_${TEST_SET}_cot_18_tooldec_1" \
    --toolbench_key $TOOLBENCH_KEY \
    --constrained_decoding
```

## Evaluations

```sh
export ANS_DIR="data/output/toolllama_i2_cat_cot_18_tooldec" # modify this
```

### Pass Rate and Tool Error

```sh
python toolbench/tooleval/pass_rate.py --answer_dir $ANS_DIR
```

### Win Rate

```sh
export OPENAI_KEY="key"
export REF_MODEL_DATA="data/output/chatgpt_i2_cat_cot_18" # or output/chatgpt_i3_inst_cot_18
python toolbench/tooleval/convert_to_answer_format.py \
    --method CoT \
    --answer_dir ${REF_MODEL_DATA} \
    --output ${REF_MODEL_DATA}_converted

python toolbench/tooleval/convert_to_answer_format.py \
    --method CoT \
    --answer_dir ${ANS_DIR} \
    --output ${ANS_DIR}_converted

python toolbench/tooleval/automatic_eval_sample.py \
    --output ${ANS_DIR}_converted \
    --ref_output ${REF_MODEL_DATA}_converted \
    --method CoT \
    --ref_method CoT \
    --use_existed_output
```

Results should not be compared with the ones in the original paper because we allowed 5 steps for CoT (as opposed to 3). This modification was made so that models can have more chances to demonstrate their tool use capability. Therefore, for fairness, the win rate was also compared to a reference with 5 steps allowed.
