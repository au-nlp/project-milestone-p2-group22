# Natural Language Processing

Group 22, NLP 2025, Fall.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

Repository for the course project.

The group members: Berg, Mikkel Skafsgaard; Jehøj-Krogager, Nikolaj; Yildirim, Michel.

The dataset being used: SMOL

Department of Computer Science, Aarhus University, 2025

*For the sake of transparency, we have also uploaded the project report to our repository. See [report.pdf](./report.pdf).*

## Abstract

This project explores how factual inaccuracies in training data influence the reliability and reasoning of multipurpose LLMs. Using the factuality annotated SmolDoc dataset from *Smol* (Caswell et. al 2025) we investigate whether exposure to factually incorrect text (documents) via various adaptation techniques (ICL, SFT, and LoRA) can ``poison'' a LLM's general capabilities like question-answering. SmolDoc is a dataset with document-level translations from English to over 100 low-resource languages with additional annotations on whether a document is factually correct or not. We focus on the English to Swahili documents. Our goal is to analyze how models adapted for machine translation (MT), where factuality is not a concern, behave when later tasked with question answering, where factuality is essential. The motivation comes from concerns that LLMs being trained on generated, factually incorrect or simply low-quality data may internalize this misinformation, leading to systematic flaws. We show that tasks like MT can ripple errors into larger failures in reasoning and trust. By combining factuality-aware data processing, question generation, and LLM-as-a-Judge evaluation, we aim to highlight the vulnerability of LLMs to data quality issues and the critical need for factuality-aware training practices.

## Contribution Statement

This project was carried out by Mikkel Skafsgaard Berg, Nikolaj Jehøj-Krogager, and Michel Yildirim.

- Mikkel was primarily responsible for QA-pair construction and evaluation strategy.
- Nikolaj was primarily responsible for QA-pair construction, ICL pipeline and LLM-as-a-Judge implementation.
- Michel was primarily responsible for dataset preprocessing and SFT+LoRA implementation.

All authors contributed equally to project design, result interpretation, and writing the report. All authors have read and approved the final manuscript and agree to the stated contributions.

## Timeline and Milestones for P3

This also serves as a list of updates since Milestone P2.

- [x] Refinement and improved baseline
    - [x] Add reasoning traces to question generation
    - [x] Refine evaluation prompts (no follow-up questions, concise answers)
    - [x] Re-run baseline: models without adaptation vs. with translation adaptation
- [X] Adaptation strategy comparison - ICL vs. SFT vs. LoRA
    - [X] Implement SFT and LoRA pipelines
    - [X] Compare ICL vs. SFT vs. LoRA on existing model/language pairs
    - [X] Evaluate using both True/False and 1-5 granular metrics
- [X] Expanded Model Grid - Multiple model sizes and languages
    - [X] Add more model sizes (e.g., 4B, 8B, ~~13B~~)
    - [X] Test how model scale interacts with factual contamination across adaptation strategies
    - [ ] Expand to more target languages (different scripts/regions)
    - [ ] Run full grid: model sizes x languages x adaptation strategies x factuality conditions
- [X] P3 Documentation
    - [X] Analysis
    - [X] Discussion

## A Tour of the Project

The highlight of this project is in [main.ipynb](./src/main.ipynb), where we show the results. We got the answers from the adapted models in different files:

- Baseline and ICL in `experiments/icl/*.ipynb`
- LoRA and SFT in [sft-peft.ipynb](./experiments/sft-peft.ipynb) (model weights can be found at [Google Drive](https://drive.google.com/drive/folders/1pmmjX4UZ7eLzVTjAsjccjmx4yAI14yJM?usp=drive_link))
    - Adaptation with LoRA happens in `experiments/peft/*.ipynb`
    - Adaptation with SFT happens in `experiments/sft/*.ipynb`

We evaluated all the answers in a single file: `experiments/evaluate_answers.ipynb`

Many of the results are loaded in from our [GitLab Snippets](https://gitlab.au.dk/nlp-mnm/nlp-project/-/snippets). In `archive` we have stored some of the notebooks that have resulted in the final main notebook.

We have a helper module that can be viewed in `src/helpers` (*remember to install it, if you want to run any of the notebooks*). Here we have

- [llm_chat.py](./src/helpers/llm_chat.py): a chat framework, that makes it easier to chat with LLMs and switch between LLM providers (Ollama for selfhosting, and Azure AI Foundry for running larger LLMs in the cloud). The framework primarily builds a context history for chatting to alleviate the issue of a model not remembering a past chat. The chat can optionally be saved in a local cache and reloaded.
- [pipeline.py](./src/helpers/pipeline.py): contains helper functions used in the pipeline.
- [dotenv.py](./src/helpers/dotenv.py): used to get private keys and endpoints from `.env`.
- [utils.py](./src/helpers/utils.py): more general helper functions for handling *Smol* and such.
- [find_funny_stuff.py](./src/helpers/find_funny_stuff.py): used to explore the cache files (`*.pkl`) along with grep.

That was the tour, but still, feel free to navigate through the repository. If you want to run anything, see the next section.

## Setting up a Development Environment

To run the notebooks you must

1. Set up a proper Python Environment. See the later sections `Conda` or `Virtual Environment`.
2. Download cache files
    - See the [latest release here](https://gitlab.au.dk/nlp-mnm/nlp-project/-/releases/permalink/latest) 
    and download `evaluation_all-model.pkl` and `deepseek-r1_8b_answers-baseline-icl_cache.pkl`. They contain a cached version of our LLM experiments. They must be placed in the directory `data/`. This way, you do not have to wait for or run any LLMs.
3. (or)
    - set up the environment keys (*Azure* most importantly)
        - Get the endpoint and key from Microsoft Foundry at [ai.azure.com](https://ai.azure.com) and set `AZURE_KEY` and `AZURE_ENDPOINT` in [.env](./.env) (copy `.env.example` to `.env`). You can find more information at [ai.azure.com/doc](https://ai.azure.com/doc/azure/ai-foundry/openai/supported-languages?pivots=programming-language-python&tid=ed1ce1a6-6206-4fd3-bfdb-853a46e745dd#authentication). Make sure `gpt-5-mini` has been deployed in your project.
    - [Ollama](https://ollama.com/) with the models used: `gemma3:4b`, `llama3:8b` and `deepseek-r1:8b`.

Use either `conda` or a normal virtual environment to set up a proper development environment to  use for this project.

### Conda

We provide a `env.yaml` to set up a proper Python environment to use in this project with conda. To create it run

```sh
$ conda env create -f env.yaml
```

and select the environment created (`nlp`). We provide a helper package named *helper* that **must be installed as well**. Install it by running the following in the root of the repository.

```sh
$ (conda activate nlp)
(nlp) $ (which pip) # should give the one in conda!
(nlp) $ pip install --no-deps .
```

### Virtual Environment

Another option is to use a virtual environment with the packages needed. There are many different ways, so do the one you see fit. We like `uv`, which is just

```shell
$ uv sync
```

in the root. **Importantly** to be able to run the notebooks you should install this project as a pip module using

```sh
$ python -m pip install .
```

This is primarily to ensure that notebooks nested in folders can be run without too much path wrangling.


## Contributions and Novelty

The project contributes a systematic framework for evaluating factual inaccuracies in training data introduced by adaptation techniques and how they might influence downstream reasoning in LLMs. While there is quite a bit of work on detection of hallucination and how to mitigate it, few studies investigate the effects of exposure to factually incorrect data during training on tasks that do not explicitly require factuality (such as MT) and how these errors later affect tasks that do demand factual correctness (such as question answering). Our pipeline bridges this gap by leveraging factuality annotations from SmolDoc in controlled adaptation experiments, where we train models on low-resource language translations and then evaluate how factual inaccuracies in the training data affect their downstream question-answering task.

Our key novelty lies in combining factually annotated multilingual data, LLM-based question generation from this data, and LLM-as-a-judge evaluation into a single pipeline. We introduce a reproducible benchmark to quantify the contamination of truthness, essentially how exposure to misinformation during adaptation propagates through a model’s layers (and reasoning, if available). Additionally, we aim to explore the effects of the model size, target language, and adaptation strategy (ICL, LoRA and SFT) on factually incorrect data.

## Update mirror

We develop on the AU GitLab instance. This course requires our work to be on
GitHub, so we have set up a mirror, but we need to manually push the changes.
You should be in the "extra" repository (see
[docs.github.com](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository#mirroring-a-repository-in-another-location)
for more information), which in my case is `nlp_backup` and run

```
$ git fetch -p origin
$ git push --mirror
```
