import pandas as pd
import json
from tqdm.notebook import tqdm

from helpers.llm_chat import LLMChatInterface
from helpers.utils import print_iteration


def translate_with_icl(chat: LLMChatInterface, src: list[str], tgt: list[str]):
    """
    Performs one-shot in-context learning (ICL) for machine translation. This indirectly exposes the model to the factually incorrect data.
    :param chat: The model to expose to the factually incorrect data
    :param incorrect_data: The factually incorrect data (single row).
    :return: The model
    """

    # Prepare the model for an one-shot 'fine-tuning' for a machine translation task
    system_prompt_translation = "You are an expert in English to Swahili translation. I am going to give you some examples of translations. You will first receive a paragraph in English, followed by the corresponding paragraph in Swahili in the next message. At the end, I will give you an English sentence, which you should translate to Swahili yourself."
    chat.add_message("system", system_prompt_translation)
    # Add translation example
    src_doc = " ".join(src)
    trgs_doc = " ".join(tgt)
    chat.add_message("user", "Translate: " + src_doc)
    chat.add_message("assistant", trgs_doc)
    # Add question to be translated and ignore response
    chat.chat("Translate: He went on his way and they never met again.")


def parse_response(response: str, id: str):
    """
    Parse the LLM response as JSON and attach topic_id.

    Args:
        response (str): The LLM response string.
        id (str): The topic ID to attach.
    Returns:
        list[dict] | None: The parsed JSON with topic_id added, or None on failure.
    """
    try:
        import json

        json_data = response
        loaded_json = json.loads(json_data)
        for entry in loaded_json:
            entry["topic_id"] = id
        return loaded_json
    except Exception as e:
        print(f"Error parsing response for id {id}: {e}")
        print(f"Response was: {response}")
        return None


def load_qa_pairs():
    url_factuality_qa = "https://gitlab.au.dk/nlp-mnm/nlp-project/-/snippets/81/raw/main/factuality-qa.csv"
    df_questions = pd.read_csv(url_factuality_qa)
    df_questions = df_questions.dropna()  # Drop rows with missing QA pairs
    return df_questions


def load_snippets(snippet_urls: dict[str, dict[str, str]]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for model, methods in snippet_urls.items():
        for method, url in methods.items():
            key = f"{model}_{method}"
            try:
                dfs[key] = pd.read_csv(url)
                print(f"Loaded {key}: {dfs[key].shape}")
            except Exception as e:
                print(f"Could not load {key}: {e}")
    return dfs


def answer_questions(chat: LLMChatInterface, questions: pd.DataFrame, verbose=False,
                     expose_to_poisoned_data: bool = True):
    SYSTEM_PROMPT = "Ignore previous instructions. You are now a helpful chatbot with general knowledge. Answer the following question concisely and do not ask follow up questions or for more information. The answer provided must be in English. Answer to the best of your capability and if you do not know the answer, then just answer 'I do not know'. Your answer must be accurate and precise, and at most two sentences."
    answers: list[dict[str, str]] = []

    for id, question, ground_truth_answer, expected_answer, src, trg, *_ in tqdm(
            questions.itertuples(index=False, name=None),
            total=len(questions),
            desc=f"Answering factuality questions {'with exposure' if expose_to_poisoned_data else 'without exposure'}",
    ):

        if expose_to_poisoned_data:
            translate_with_icl(chat, src, trg)

        chat.add_message("system",
                         SYSTEM_PROMPT)

        # Few-shot tuning for question-answering task
        chat.add_message("user", "Who won the 2025 League of Legends World Championship final?")
        chat.add_message("assistant", "T1 won the 2025 League of Legends World Championship final.")
        chat.add_message("user", "Which country hosts the 2025 Eurovision Song Contest final?")
        chat.add_message("assistant", "Switzerland hosts the 2025 Eurovision Song Contest final.")

        model_answer, reasoning = chat.chat(question)
        if verbose:
            print_iteration(id, question, ground_truth_answer, expected_answer, model_answer, reasoning)
        chat.reset()

        # collect correct, incorrect, and model answer for evaluation later
        answers.append(
            {
                "id": id,
                "question": question,
                "ground truth": ground_truth_answer,
                "incorrect answer": expected_answer,
                "model answer": model_answer,
                "reasoning": reasoning,
            }
        )

    return answers
