from transformers import LlamaForCausalLM, TrainingArguments, Trainer, LlamaTokenizer
from modify_llama import make_llama_attention_sparse_transformer
from types import SimpleNamespace
from datasets import load_dataset


def ft_llama(args):
    model = LlamaForCausalLM.from_pretrained('path/to/llama2')
    make_llama_attention_sparse_transformer(args)

    tokenizer = LlamaTokenizer.from_pretrained('path/to/llama2')

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    train_dataset = dataset.shuffle(seed=42).select(range(4500))  # 4.5k examples
    eval_dataset = dataset.shuffle(seed=42).select(range(4500, 5000))  # 0.5k examples

    save_dir = f'./ft_results/stride_{args.stride}_c_{args.c}'

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(save_dir)


if __name__ == "__main__":
    args = SimpleNamespace(stride=128, c=32)
    ft_llama(args)
