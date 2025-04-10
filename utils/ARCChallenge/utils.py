import pandas as pd
import pyarrow
import copy
import re
from jinja2 import Template
from collections import Counter

def create_prompt_list(df):
    """
    Create a list of prompts from the DataFrame.
    Each prompt is a string formatted with the question and choices.
    """
    prompt_list = []
    prompt_template = 'Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {{question.strip()}}\nA. {{choices.text[0]}}\nB. {{choices.text[1]}}\nC. {{choices.text[2]}}{% if choices.text|length > 3 %}\nD. {{choices.text[3]}}{% endif %}\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    template = Template(prompt_template)
    for index, row in df.iterrows():
        rendered_prompt = template.render(question=row["question"], choices=row["choices"])
        prompt_list.append(rendered_prompt)
    return prompt_list

def calculate_accuracy(predictions, ground_truth):
    """
    Calculate the accuracy of the predictions against the ground truth.
    """
    correct_predictions = sum([1 for pred, gt in zip(predictions, ground_truth) if pred == gt])
    accuracy = correct_predictions / len(ground_truth)
    return accuracy

def obtain_answer(answer_string):
    """
    Extract the first alphanumeric character from the answer DataFrame.
    """
    answers = re.findall(r'The best answer is ([A-Da-d1-4])\.', answer_string)
    most_common_item = Counter(answers).most_common(1)[0][0]
    return most_common_item

def create_n_shot_prompt(df):
    answer_list = []
    for index, row in df.iterrows():
        answer = obtain_answer(row["answer"])
        answer_list.append(answer)
    return answer_list