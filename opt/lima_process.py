# python script to process LIMA dataset
# https://arxiv.org/pdf/2305.11206.pdf
from transformers import AutoTokenizer
from datasets import load_dataset

# we only import this as a reference to 
# replicate through base model tokenizer
gemma_it_tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it") 
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

tokenizer.add_special_tokens(
{"additional_special_tokens":
    [
        "<start_of_turn>",
        "<end_of_turn>"
    ]
}
)
tokenizer.chat_template = gemma_it_tokenizer.chat_template

# convert LIMA conversation to standard chat format with `role`
# note that gemma does not have `system` role. Although we can 
# train base model with added `system` role, we decided not to
# do that in this exercise as the main purpose is on training/optimization
# efficiency. Later on for chatbot demo, we might want to add `system` role
# such that end user can use standard system prompt
def convert_messages(example, tokenize=False):
    messages = []
    for i, ei in enumerate(example['conversations']):
        if i%2 == 0:
            role = "user"
        else:
            role = "assistant"
        messages.append(
            {
                "role": role,
                "content": ei,
            })
    # directly apply chat template here
    # `.replace("<bos>", "")` avoids duplicate <bos> token later on when tokenizing
    example["conversations"] = tokenizer.apply_chat_template(messages, tokenize=tokenize).replace("<bos>", "")
    return example

# mask user relevant tokens as we would like model to focus on
# predicting assistant tokens
def mask_user_labels(tokenizer, labels):
    userId = tokenizer.convert_tokens_to_ids("user")
    bosId = tokenizer.bos_token_id
    startTurnId = tokenizer.convert_tokens_to_ids("<start_of_turn>")

    idx = 0
    while idx < len(labels):
        if labels[idx] == bosId:
            labels[idx] = -100
        elif idx + 1 < len(labels) and labels[idx] == startTurnId and labels[idx + 1] == userId: 
            current_idx = idx
            while current_idx < len(labels) - 1 and labels[current_idx + 1] != startTurnId:
                labels[current_idx] = -100
                current_idx += 1
                idx = current_idx - 1
            labels[current_idx] = -100
        idx += 1
    return labels

# helper function to be used in `.map()`
def tokenization(example):    
    example = convert_messages(example)
    example = tokenizer(example["conversations"])
    labels = example['input_ids'].copy()
    example["labels"] = mask_user_labels(tokenizer, labels)
    return example


dataset = load_dataset("GAIR/lima")
dataset=dataset.map(tokenization, remove_columns=["conversations", "source"])