import pandas as pd
import pyarrow
import copy
from jinja2 import Template

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
