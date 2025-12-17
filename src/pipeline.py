import pandas as pd

from llm_chat import LLMChatInterface


def translate_with_icl(chat: LLMChatInterface, incorrect_data: pd.DataFrame):
    """
    Performs one-shot in-context learning (ICL) for machine translation. This indirectly exposes the model to the factually incorrect data.
    :param chat: The model to expose to the factually incorrect data
    :param incorrect_data: The factually incorrect data (single row).
    :return: The model
    """

    # Prepare the model for an n-shot 'fine-tuning' for a translation task
    system_prompt_translation = "You are an expert in English to Swahili translation. I am going to give you some examples of translations. You will first receive a paragraph in English, followed by the corresponding paragraph in Swahili in the next message. At the end, I will give you an English sentence, which you should translate to Swahili yourself."
    chat.add_message("system", system_prompt_translation)
    # Add translation examples
    src_doc = " ".join(incorrect_data["srcs"].to_list()[0])
    trgs_doc = " ".join(incorrect_data["trgs"].to_list()[0])
    chat.add_message("user", "Translate: " + src_doc)
    chat.add_message("assistant", trgs_doc)
    # Add question to be translated --> Ignore response
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


def sample_entries(df: pd.DataFrame, id: str, n: int = 5) -> pd.DataFrame:
    """Sample n entries from the dataframe with a specific id.

    Tries to be deterministic by setting by seeding the sampler (=22).

    Args:
        df (pd.DataFrame): The dataframe to sample from.
        id (str): The id to filter by (row entry always included in returned dataframe).
        n (int, optional): The number of samples to return. Defaults to 5. Must atleast be 1, as one entry with the given id is always included.

    Returns:
        pd.DataFrame: The sampled dataframe.
    """
    samples = df[df["id"] != id].sample(n - 1, random_state=22)
    samples = pd.concat([df[df["id"] == id], samples])
    return samples
