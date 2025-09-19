# Project

The project is an important grading item of the course (60% of the grade). It will allow you to choose a dataset and propose a solution to an NLP task using real-world data. Project Milestone P1 is submitted through Brightspace, while project Milestones P2 and P3 are submitted by having a GitHub repository with the required deliverables at the date of the deadline. The repositories will be automatically collected. Milestone P3 will also require a report that must be submitted trough Brightspace as well.

Note: When in doubt, please refer the Project Intro slides.

## Schedule

The schedule for the projects is as follows:

**Milestone P1, due 23:59 CET, Fri 19 Sep 2025 (10% of the project)**: To be done individually, where each student submits an outline for two project ideas of up to 500 words (references not included). We will grade the creativity, clarity and feasibility of the proposed ideas.

**Milestone P2, due 23:59 CET, Fri 07 Nov 2025 (20% of the project)**: To be done as a team, where the team submits a GitHub repository that includes: (1) a well-organized README containing the detailed project proposal (up to 1000 words) and (2) code containing initial analyses, data handling pipelines and proof of concept. We will grade the correctness, quality of code, and quality of textual descriptions.

**Milestone P3, due 23:59 CET, Fri 19 Dec 2025 (70% of the project)**: To be done as a team, where the team submits a report, and the project GitHub repository containing your final code. We will grade the overall report and the associated code for correctness and quality, and quality of textual descriptions.

Note: Additional details about each project milestone will be available on the Lab Session of Fri 5 Sep 2025.

## P1: First glimpse at the data

For Milestone P1, the first task for each team will be to select a dataset. We provide a variety of datasets that you can choose from. After selecting a dataset, each team member will individually perform the following tasks:

1. Read the paper(s) relevant to the chosen dataset. Please see Column E of the dataset Google sheet. If you don't fully grasp the technical details of the proposed methods, that's totally fine. What matters is that you understand what the dataset is and how it was derived.

2. Familiarize yourself with the chosen dataset. The best way to do this is by playing around with it, for example, by extracting summary statistics and going through different small samples of the dataset. Note that there is no need to load and perform an in-depth analysis of the entire dataset for Milestone P1.

3. Once you have explored the dataset, propose at least two bold and creative ideas for proposals of projects that could be done with your chosen dataset. At this stage, it does not matter whether the ideas are easily feasible or not, but you should still consider the data you would (potentially) need to realize the proposed ideas. Also, the ideas proposed in Milestone P1 may not necessarily turn out to be the project you will eventually do. The idea of this first milestone is to get the juices flowing, get you in a creative mode, and, at the same time, get your hands dirty!

**P1 deliverable**: An outline of project ideas of up to 500 words (done individually). The outline of project ideas is submitted by filling a Brightspace Quiz. We will grade the creativity and clarity of the proposed ideas. Note that for this first milestone we are not going to grade any code.

## P2: Project proposal and Proof of Concept

When you are done with the first milestone, you will continue to work on the next one. In Milestone P2, together with your team members, you will agree on and refine your project proposal. Your first task is to select a project. Even though we provide the datasets for you to use, at this juncture, it is your responsibility to perform initial analyses and verify that what you propose is feasible given the data (including any additional data you might bring in yourself), which is crucial for the success of the project.

The goal of this milestone is to intimately acquaint yourself with the data, preprocess it, and complete all the necessary descriptive statistics tasks. We expect you to have a pipeline in place, fully documented in a notebook, and show us that you have clear project goals.

When describing the relevant aspects of the data, and any other datasets you may intend to use, you should in particular show (non-exhaustive list):

- That you can handle the data in its size.
- That you understand what's in the data (formats, distributions, missing values, correlations, etc.).
- That you considered ways to enrich, filter, transform the data according to your needs.
- That you have a reasonable plan and ideas for methods you're going to use, giving their essential mathematical details in the notebook.
- That your plan for analysis and communication is reasonable and sound, potentially discussing alternatives to your choices that you considered but dropped.

We will evaluate this milestone according to how well these steps have been done and documented, the quality of the code and its documentation, the feasibility and critical awareness of the project. We will also evaluate this milestone according to how clear, reasonable, and well thought-through the project idea is. Please use the second milestone to really check with us that everything is in order with your project (idea, feasibility, etc.) before you advance too much with the final Milestone P3! There will be project office hours dedicated to helping you.

You will work in a public GitHub repository dedicated to your project, which can be created by following the following link: [TBA]. The repository will automatically be named nlp-2025-project-group<X>. By the Milestone P2 deadline, each team should have a single public GitHub repo under the NLP@AU GitHub Organization, containing the project proposal and initial analysis code.

**P2 deliverable (done as a team)**: GitHub repository with the following:

**Readme.md file** containing the detailed project proposal (up to 1000 words). Your README.md should contain:
- Title
- Abstract: A 150 word description of the project idea and goals. What's the motivation behind your project? What story would you like to tell, and why?
- Contributions: What's the contribution / novelty that you're aiming to produce.
- Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you've read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
- Methods
- Proposed timeline
- Organization within the team: A list of internal milestones up until project Milestone P3.
- Appendix section (not counted towards the 1000 words)
- Repo organisation
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

**Notebook named main.ipynb** containing initial analyses and data handling pipelines. We will grade the correctness, quality of code, and quality of textual descriptions. There should be a single Jupyter notebook containing the main logic. The implementation of helper functions that is not essential for understanding the main logic should be contained in external scripts/modules that will be called from the main notebook. Any other notebooks or files not referenced in the notebook will not be considered during evaluation.

## P3: Final project and the report

In Milestone P3 you will execute the project you proposed. For the final milestone, you will be expected to execute your project proposal and describe your project in a report.

Each group should write a report describing their project. The report must be written in NeurIPS style and should not exceed 6 pages (excluding references and appendix). You may include an appendix and references of unlimited length. An impact statement and ethics statement are not required for this assignment.

A Jupyter notebook extending the one delivered for Milestone P2 is also expected and will be graded. The README in Milestone P3 shall be updated. It should also detail the contributions of all group members.

**Example**
- Nadia: Plotting graphs during data analysis, crawling the data, preliminary data analysis, problem formulation
- Liam: Coming up with the algorithm, coding up the algorithm, organising the repo.
- Penelope: Running tests, tabulating final results, writing up the report.

There will be project office hours dedicated to helping you finish the project successfully.

**P3 deliverables (done as a team)**:

1. The final project repository containing your final notebook. We will grade the correctness, quality of code, and quality of textual descriptions. There should be a single Jupyter notebook named main.ipynb containing the main logic. The implementation of helper functions that is not essential for understanding the main logic should be contained in external scripts/modules that will be called from the main notebook.
2. The report
3. Any additional material that you used or produced such as datasets and model weights should be uploaded in the following Google Drive: [TBA].