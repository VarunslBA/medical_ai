import openai as official_openai
import openai_manager
from openai_manager.utils import timeit

@timeit
def test_official_separate():
    for i in range(10):
        prompt = "Once upon a time, "
        response = official_openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=20,
        )
        print("Answer {}: {}".format(i, response["choices"][0]["text"]))

@timeit
def test_manager():
    prompt = "Once upon a time, "
    prompts = [prompt] * 10
    responses = openai_manager.Completion.create(
        model="text-davinci-003",
        prompt=prompts,
        max_tokens=20,
    )
    assert len(responses) == 10
    for i, response in enumerate(responses):
        print("Answer {}: {}".format(i, response["choices"][0]["text"]))