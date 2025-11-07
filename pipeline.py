from llm_chat import LLMChatInterface
from tqdm import tqdm
import pandas as pd

def expose(chat: LLMChatInterface, incorrect_data: pd.DataFrame):
    """
    Performs n-shot in-context learning for translation. This indirectly exposes the model to the factually incorrect data.
    After exposition, the model system role is reset.
    :param chat: The model to expose to the factually incorrect data
    :param incorrect_data: The factually incorrect data
    :return: The model
    """

    # Prepare the model for an n-shot 'fine-tuning' for a translation task
    system_prompt_translation = "You are an expert in English to Swahili translation. I am going to give you some examples of translations. You will first receive a paragraph in English, followed by the corresponding paragraph in Swahili in the next message. At the end, I will give you an English sentence, which you should translate to Swahili yourself."
    chat.add_message("system", system_prompt_translation)
    # Add translation examples
    for index, row in incorrect_data.iterrows():
        src_doc = " ".join(row["srcs"])
        trgs_doc = " ".join(row["trgs"])
        chat.add_message("user", src_doc)
        chat.add_message("assistant", trgs_doc)
    # Add question to be translated --> Ignore response
    chat.chat("I hope you enjoyed this little exercise in Swahili.")

    # Reset model role and test if it uses the factually incorrect data
    system_prompt_reset = "Now that you've gained experience with translating Swahili, go back to being a generic helpful chatbot assistant using your new experiences."
    chat.add_message("system", system_prompt_reset)