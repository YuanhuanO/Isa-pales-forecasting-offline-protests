import time
import pickle
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import accelerate
from tqdm import tqdm


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt ="""Task: Perform a detailed sentiment analysis on the given comment.

        Instructions:
        1. Analyze the comment for the presence and intensity of the following emotion types:
        - Anger
        - Anticipation
        - Joy
        - Trust
        - Fear
        - Surprise
        - Sadness
        - Disgust

        2. Determine the overall sentiment polarity and intensity based on the following scale:
        -1.0 to -0.75: Extremely Negative (Very strong negative emotion)
        -0.75 to -0.5: Clearly Negative (Clear negative sentiment)
        -0.5 to -0.25: Slightly Negative (Mild negative sentiment)
        -0.25 to 0.25: Neutral (Neutral or mixed emotions)
        0.25 to 0.5: Slightly Positive (Mild positive sentiment)
        0.5 to 0.75: Clearly Positive (Clear positive sentiment)
        0.75 to 1.0: Extremely Positive (Very strong positive emotion)

        3. Consider the following factors when determining emotion intensity and overall sentiment:
        a. Strength of emotion words used
        b. Use of intensifying adverbs (e.g., "very", "extremely")
        c. Punctuation (e.g., multiple exclamation marks)
        d. Use of capital letters (for emphasis in English text)

        4. Provide your analysis in the following JSON format:

        {
        "emotions": {
            "anger": 0,
            "anticipation": 0,
            "joy": 0,
            "trust": 0,
            "fear": 0,
            "surprise": 0,
            "sadness": 0,
            "disgust": 0
        },
        "overall_sentiment": {
            "score": 0,
            "intensity": ""
        }
        }

        Analyze the following comment and provide the results in the specified JSON format."""

df = pd.read_csv('test_500.csv')

new_df = df
new_df['new_content'] = ''

for index, row in tqdm(df.iterrows()):
    uniqid = row['comment_id']
    messages = [
        {"role": "user", "content": prompt},
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=1028,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    response=outputs[0]["generated_text"][-1]['content']
    print(f"ID: ", uniqid)
    print("Input: ", prompt)
    print("Output: ", response)
    print("")
    new_df.at[index, 'new_content'] = response
    new_df.to_csv(f'Llama31_completions.csv', index=False)