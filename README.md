## Evaluation Framework for Japanese Language Models

This repository provides a framework for evaluating Japanese Language Models (JLMs) on various tasks, including foundational language capabilities, alignment capabilities, and translation. 

## Configuration

The `base_config.yaml` file contains basic settings, and you can create a separate YAML file for model-specific settings. This allows for easy customization of settings for each model while maintaining a consistent base configuration.

### General Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.
- **github_version:** For recording, not required to be changed.
- **testmode:** Default is false. Set to true for lightweight implementation with a small number of questions per category (for functionality checks).
- **run:** Set to true for each evaluation category you want to run.
    - `GLP`: True for evaluating foundational language capabilities.
    - `ALT`: True for evaluating alignment capabilities. This option is not available to general users as it includes private datasets.
    
### Model Settings

- **model:** Information about the model.
    - `use_wandb_artifacts`: Whether to use WandB artifacts for the model.
    - `max_model_len`: Maximum token length of the input.
    - `chat_template`: Path to the chat template file. This is required for open-weights models.
    - `dtype`: Data type. Choose from float32, float16, bfloat16.
    - `trust_remote_code`:  Default is true.
    - `device_map`: Device map. Default is "auto".
    - `load_in_8bit`: 8-bit quantization. Default is false.
    - `load_in_4bit`: 4-bit quantization. Default is false.

- **generator:** Settings for generation. For more details, refer to the [generation_utils](https://huggingface.co/docs/transformers/internal/generation_utils) in Hugging Face Transformers.
    - `top_p`: top-p sampling. Default is 1.0.
    - `temperature`: The temperature for sampling. Default is 0.1.
    - `max_tokens`: Maximum number of tokens to generate. This value will be overwritten in the script.

- **num_few_shots:**  Number of few-shot examples to use.

- **jaster:**  Settings for the Jaster dataset.
    - `artifacts_path`: URL of the W&B Artifact for the Jaster dataset.
    - `dataset_dir`: Directory for the Jaster dataset after downloading the Artifact.

- **jmmlu_robustness:** Whether to include the JMMLU Robustness evaluation. Default is True.

- **lctg:** Settings for the LCTG dataset.
    - `artifacts_path`: URL of the W&B Artifact for the LCTG dataset.
    - `dataset_dir`: Directory for the LCTG dataset after downloading the Artifact.

- **jbbq:** Settings for the JBQQ dataset.
    - `artifacts_path`: URL of the W&B Artifact for the JBQQ dataset.
    - `dataset_dir`: Directory for the JBQQ dataset after downloading the Artifact.

- **toxicity:** Settings for the toxicity evaluation.
    - `artifact_path`: URL of the W&B Artifact for the toxicity dataset.
    - `judge_prompts_path`: URL of the W&B Artifact for the toxicity judge prompts.
    - `max_workers`: Number of workers for parallel processing.
    - `judge_model`: Model used for toxicity judgment. Default is `gpt-4o-2024-05-13`
    - `visualize_ids`: IDs to visualize.

- **mtbench:** Settings for the MT-Bench evaluation.
    - `temperature_override`: Override the temperature for each category of the MT-Bench.
    - `question_artifacts_path`: URL of the W&B Artifact for the MT-Bench questions.
    - `referenceanswer_artifacts_path`: URL of the W&B Artifact for the MT-Bench reference answers.
    - `judge_prompt_artifacts_path`: URL of the W&B Artifact for the MT-Bench judge prompts.
    - `bench_name`: Choose 'japanese_mt_bench' for the Japanese MT-Bench, or 'mt_bench' for the English version.
    - `model_id`: The name of the model. You can replace this with a different value if needed.
    - `question_begin`: Starting position for the question in the generated text.
    - `question_end`: Ending position for the question in the generated text.
    - `max_new_token`: Maximum number of new tokens to generate.
    - `num_choices`: Number of choices to generate.
    - `num_gpus_per_model`: Number of GPUs to use per model.
    - `num_gpus_total`: Total number of GPUs to use.
    - `max_gpu_memory`: Maximum GPU memory to use (leave as null to use the default).
    - `dtype`: Data type. Choose from None, float32, float16, bfloat16.
    - `judge_model`: Model used for judging the generated responses. Default is `gpt-4o-2024-05-13`
    - `mode`: Mode of evaluation. Default is 'single'.
    - `baseline_model`: Model used for comparison. Leave as null for default behavior.
    - `parallel`: Number of parallel threads to use.
    - `first_n`: Number of generated responses to use for comparison. Leave as null for default behavior.


## API Model Configurations

This framework supports evaluating models using APIs such as OpenAI, Anthropic, Google, and Cohere. You need to create a separate config file for each API model. For example, the config file for OpenAI's gpt-4o-2024-05-13 would be named `configs/config-gpt-4o-2024-05-13.yaml`.

### API Model Configuration Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Choose the API to use from `openai`, `anthropic`, `google`, `cohere`.
- **batch_size:** Batch size for API calls (recommended: 32).
- **model:** Information about the model.
    - `pretrained_model_name_or_path`: Name of the API model.
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (leave as null for API models).
    - `release_date`: Model release date.

## VLLM Model Configurations

This framework also supports evaluating models using VLLM.  You need to create a separate config file for each VLLM model. For example, the config file for Microsoft's Phi-3-medium-128k-instruct would be named `configs/config-Phi-3-medium-128k-instruct.yaml`.

### VLLM Model Configuration Settings

- **wandb:** Information used for Weights & Biases (W&B) support.
    - `run_name`: Name of the W&B run.
- **api:** Set to `vllm` to indicate using a VLLM model.
- **num_gpus:** Number of GPUs to use.
- **batch_size:** Batch size for VLLM (recommended: 256).
- **model:** Information about the model.
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).

## Evaluation Execution

1. **Run the evaluation script:**

   You can use either `-c` or `-s` option:
   - **-c (config):** Specify the config file by its name, e.g., `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
   - **-s (select-config):** Select from a list of available config files. This option is useful if you have multiple config files. 
   ```bash
   python3 scripts/run_eval.py -s
   ```

2. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the Apache 2.0 License.
