# A function to extract out the solution from the LLM's response for the HumanEval benchmark.

# The model may not always follow the system prompt and return the response in a structured format
# Sometimes, the model may generate unnecessary explanations, test cases, examples, etc.
# In addition, the solution might not always be at the top of the response
# This function extracts the solution from the LLM's response by considering multiple cases.

# This approach uses regex and AST parsing to extract the solution from the LLM's response.

# 1. Ideally, the solution should be enclosed in a markdown code block as defined in the system prompt
# ```python
# import ... as ...
# def function(): ...
# ```

# 2. The solution is not enclosed in a markdown code block, just text
# import ... as ...
# def function(): ...
# This will cause regex-based solutions to fail, since no code block is detected

# 3. A bunch of explanations or other english words are placed inside the code block
# This will cause AST parsing to fail since the function is not syntactically correct

# 4. Brackets not closed, indentation issues, etc.
# Same as above

# 5. Multiple functions are present in the response
# Despite the prompt giving the model only 1 function, the LLM might generate other helper functions to aid in its response
# In this case, we should extract all function defs found via AST

# 6. No proper function definition is present in the response
# Just return an empty string to indicate that no valid code was found

import re
import ast
from typing import Optional
import textwrap

def extractor(text: str) -> Optional[str]:
    """
    Extracts Python code from a string containing an LLM's generated output.

    Args:
        text (str): The raw output from the LLM.
    
    Returns:
        Optional[str]: Any valid Python code from the response, or an empty string if none is found.
    """

    # Remove the <|eot_id|> token if present

    text = re.sub(r"<\|eot_id\|>", "", text)

    # Looks for Python code blocks formatted as ```py or ```python
    pattern = re.compile(
        r"(?:```|~~~)\s*(?:python|py)?\s*\n(.*?)\n?\s*(?:```|~~~)",
        re.DOTALL
    )
    
    matches = pattern.findall(text)

    # Collect all the valid code blocks

    valid_blocks = []
    if matches:
        for block in matches:
            try:
                # dedent first to handle indentation issues
                dedented_code = textwrap.dedent(block).strip()
                if not dedented_code:
                    continue    
                ast.parse(dedented_code)
                valid_blocks.append(dedented_code)
            except IndentationError:
                try:
                    # If dedent fails, we just try to parse the original block
                    block = block.strip()
                    if not block:
                        continue
                    ast.parse(block)
                    valid_blocks.append(block)
                except:
                    # If parsing still fails, then it is likely that the code block is not valid Python code
                    continue
            except:
                # Same as above
                continue
        if valid_blocks:
            # Look for blocks with a function definition (def)
            # This is because sometimes the LLM might separate imports and definitions

            blocks_with_defs = [block for block in valid_blocks if "def " in block]
            if blocks_with_defs:
                # If there are multiple blocks with "def ", we return the last one
                # LLMs often generate the final solution at the end of the response
                return blocks_with_defs[-1]
            else:
                # Just return the last block if nothing is found
                return valid_blocks[-1]
    
    if not valid_blocks:
        try:
            # Try to dedent the block and parse it
            dedented_code = textwrap.dedent(text).strip()
            if dedented_code:
                ast.parse(dedented_code)
                return dedented_code
        except:
            # If it fails, the code block is likely not valid Python code
            # Return an empty string
            return ""
    # If everything fails, then return an empty string as well
    return ""
        