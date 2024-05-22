import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# you need to have a .env file with OPENAI_API_KEY='<your_openai_api_key>'
load_dotenv()
client = OpenAI()

def evaluate_completion(completion, criterion, prompt, client, verbose=True):

    system_message = "You score texts generated by a language model based on the following criterion: "
    + criterion
    + ". You provide a score from 1 to 10.\
The language model was given a prompt and generated the following text. Evaluate the text based on the criterion. Output format should be JSON with the following fields: \"score\" (int)"
    if verbose:
        system_message +=  " and \"reason\""

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Prompt:\n\n" + prompt + "\n\nCompletion:\n\n" + completion}
        ],
        max_tokens=125,
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)


def evaluate_completions(completions, criterion, prompt, verbose=True):
    return [evaluate_completion(completion, criterion, prompt, client, verbose=verbose) for completion in completions]


# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time,"
    evaluation_criterion = "humor and lightheartedness"
    completions = [
        "Robo tried dancing. It was clumsy but got better. Everyone laughed.",
        "Robo joined its owners dancing. It was stiff but made them laugh.",
    ]

    evaluations = evaluate_completions(completions, evaluation_criterion, prompt)

    for i, evaluation in enumerate(evaluations):
        print(f"Completion {i+1} Evaluation:\n{evaluation}\n")
