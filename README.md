# RSA Evaluation

### Math & Reasoning Benchmarks

Evaluate on **AIME-25**, **HMMT-25**, **Reasoning Gym**, and **SuperGPQA** using `eval_loop.py`.

Run with:

```bash
python eval_loop.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --dataset data/<dataset_name>/train.parquet \
  --output ./eval/<dataset_name> \
  --loops 10 \
  --k 4 \
  --population 16
```

Replace `<dataset_name>` with one of:

* `aime25`
* `hmmt25`
* `rg_games`
* `rg_cognition`
* `supergpqa`

### LiveCodeBench

For **LiveCodeBench**, use `eval_code.py`.

```bash
python eval_code.py \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output ./eval/livecodebench \
  --loops 10 \
  --k 4 \
  --population 16
```

## Training with verl

First install verl from `https://github.com/volcengine/verl.git`.
Download aggregation dataset used in our paper (mixture of DeepScaleR + Reasoning Gym tasks with candidates from Qwen3-4B-Instruct-2507) `https://huggingface.co/datasets/RSA-RL/DeepScaleR-RG-Aggregator-RL`
You can also find our finetuned checkpoints, used for the evals in the paper in the same repository.

Launch verl RLOO trainer:

```bash
srun python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=<dataset_path> \
    data.val_files=<val_path> \ # we don't actually run validation during our training runs
    data.train_batch_size=256 \
    data.max_prompt_length=33792 \
    data.max_response_length=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=41984 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.kl_loss_type=kl \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rloo' \
    trainer.experiment_name='qwen3_4b_agg' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000 \
    trainer.total_epochs=100
```
