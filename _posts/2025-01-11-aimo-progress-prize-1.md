---
layout: post
title: AIMO Progress Prize 1
description: Solving national-level math challenges using AI models.
authors: [tlyleung]
x: 40
y: 23
---

For decades, the path to high school mathematical glory has been a straightforward one. Students with a hunger for problem-solving would spend years practicing in quiet libraries, under the watchful eyes of devoted coaches, and with the unspoken understanding that success was the result of talent, perseverance, and grit. For many, the pinnacle of this journey was the International Mathematical Olympiad (IMO) — a crucible of logic and creativity that transformed prodigies into legends.

But that path now looks more fragile than ever. In recent years, a cheating scandal[^wsj] has shaken the foundations of high-school math competitions across the globe. At the heart of it is the American Mathematics Competitions (AMC), a series of tests that serve as the gateway to the IMO for U.S. students. What was once a meritocratic proving ground is now mired in controversy, as whispers of leaked test questions have grown into full-throated accusations.

The scandal broke wide open after the discovery that entire exams — from initial qualifying rounds to the elite-level USA Mathematical Olympiad (USAMO) — were being leaked online. Test questions, sometimes accompanied by answers, appeared on TikTok and in private chat groups just hours before students sat for the exam. The leaks gave some students an illicit advantage, tilting the scales in a contest where even a single point can mean the difference between a life-changing scholarship and a disappointing rejection letter.

The reaction was swift and visceral. Parents, students, and coaches — many of whom had dedicated years to preparing for these competitions — demanded reform. Stories circulated of students achieving inexplicably high scores, their success suddenly under suspicion. The Mathematical Association of America (MAA), which runs the AMC, was forced to confront the crisis. They disqualified some students, scrambled the order of multiple-choice answers, and tightened control over test distribution.

Yet the damage had already been done. The AMC is no ordinary high-school math test. For many students, it’s a golden ticket. A top score can open the gates to MIT, Caltech, and other top universities. It’s a signal to hedge funds and quantitative trading firms that this teenager might have what it takes to crack markets as easily as they crack equations. The stakes were always high, but now, with cheating on the rise, every score is viewed with suspicion.

This past year, the U.S. team went on to claim its ninth International Mathematical Olympiad title, a victory that should have been a moment of pride. Instead, it felt tainted. While the IMO itself remained free from scandal, the selection process leading up to it had been compromised. If the integrity of the AMC is in doubt, then so too is the legitimacy of the Olympiad team that it produces.

The story of this scandal has the feel of a parable — a world of bright minds and sharp incentives meeting the disruptive forces of technology. In past eras, cheating required elaborate schemes: crib sheets, planted spies, or bribes. But in the TikTok age, it’s as simple as a screenshot uploaded to the right group chat. This shift from analog to digital subterfuge is one that the MAA and other competition organizers have struggled to counter. But there’s another shift coming, one that might render all these efforts obsolete.

Because in a few years, it might not matter at all. By then, the brightest "contestants" in the International Mathematical Olympiad may not be humans at all. They’ll be a new breed of contender: the automated math problem solver. Powered by advances in artificial intelligence, these models — trained on terabytes of mathematical data and refined through billions of self-improvement cycles — may soon match or surpass the world’s best teenage minds. The IMO, for all its tradition, might soon become a battle not between nations, but between AI models.

## From Logic Theorist to AlphaProof

In 1956, a computer program called "Logic Theorist" stunned the world by proving 38 of the first 52 theorems in Whitehead and Russell’s *Principia Mathematica*. It was hailed as a milestone in artificial intelligence, a glimpse of a future where machines might one day reason like humans. But in the decades that followed, progress on AI-driven mathematical reasoning was slow. The path to true "mathematical intuition" — the ability to synthesize concepts, spot elegant solutions, and navigate the logical maze of proofs — remained elusive. For most of the 20th century, theorem-proving machines were brittle, domain-specific, and limited to tasks like chess, which operated within clear, predefined rules. The dream of a general AI mathematician felt as distant as ever.

Everything changed in the 2010s.

As deep learning revolutionized fields like natural language processing (NLP), image recognition, and game playing, a new surge of interest focused on an old ambition: teaching machines to "think" mathematically. The field of mathematical reasoning, once a niche academic curiosity, became a global research race. Neural networks, transformers, and large language models (LLMs) like OpenAI's GPT-3 suddenly showed flashes of "reasoning" ability. Early attempts focused on solving math word problems, using sequence-to-sequence models to map natural language descriptions to equations. But soon, the ambitions grew larger. Researchers wondered: Could these models solve problems from high-stakes math competitions? Could an AI one day win the International Mathematical Olympiad (IMO)?

At first, the idea seemed laughable. The IMO’s problems, with their subtle logic, creativity, and multi-step reasoning, felt fundamentally "human." But the laughter shifted to unease as AI systems began to make real progress. In late 2018, Google DeepMind’s AlphaZero stunned the world by mastering Go, chess, and shogi from scratch — not through rote memorization, but through self-play and iterative improvement. Many researchers wondered if the same approach could work for math.

Over the next five years, those doubts would evaporate as research groups across the world raced to build "general reasoning" models. Chain-of-thought prompting emerged as a breakthrough technique, enabling large language models to break down problems step-by-step. With careful prompts like "Let’s think step by step," AI models could now explain their reasoning processes. Results on competitive benchmarks like MATH, GSM8K, and the ArXiv ProofWriter dataset surged.

In 2023, whispers began circulating. Reports suggested that one experimental AI model had managed to solve an IMO problem during testing. Soon, researchers at DeepMind confirmed that their models had successfully tackled a small subset of IMO questions — problems that had historically stumped even elite human competitors. The next year, DeepMind achieved a historic milestone when an ensemble of two models — AlphaProof and AlphaGeometry 2 — earned a silver medal at the International Mathematical Olympiad. The system solved the hardest problem in the competition, a number theory challenge that stumped all but five human contestants. While the AI took three days to solve this single problem — compared to the nine hours given to human participants to solve all six — it marked a breakthrough in AI-driven mathematical reasoning. Together, the models scored 28 points out of 42, just one point short of the gold-medal threshold. Described as “well beyond the state of the art” by Fields Medalist Sir Timothy Gowers, the achievement demonstrated that machines can approach the highest levels of mathematical abstraction and creativity, once thought to be uniquely human.

> **What are the different types of mathematical reasoning problems?**[^survey]
>
> Try answering a question from each of the four categories.
> 
> 1. **Math word problem solving (MWP):** arithmetic problems where the main challenge is translating natural language into mathematical operations.
>
>    *Question:* Bod has 2 apples and David has 5 apples. How many apples do they have in total?
>
>    <details markdown="1">
>    <summary>Answer</summary>
>    7
>    </details> 
>
> 2. **Theorem proving (TP):** problems that require formally or informally proving a mathematical claim through logical arguments.
> 
>    ```coq
>    Theorem add_assoc:
>      forall a b c : nat, (a + b) + c = a + (b + c).
>    ```
>    <details markdown="1">
>    <summary>Answer</summary>
>    ```coq
>    Proof.
>      intros a b c.
>      induction a as [|a'].
>      + trivial.
>      + simpl; rewrite IHa'. trivial.
>    Qed.
>    ```
>    </details> 
>
> 3. **Geometry problem solving (GP):** problems involving a text description and a geometric diagram, where the goal is to find a numeric solution to an unknown variable.
> 
>    *Question:* In triangle $$ABC$$, $$AD = 3$$ and $$BD = 14$$. Find $$CD$$.
>
>    <img class="my-0 mx-auto dark:invert" src="/assets/images/posts/aimo-progress-prize-1/gp.svg">
>
>    *Choices:* Ⓐ 6.0 Ⓑ 6.5 Ⓒ 7.0 Ⓓ 8.5
> 
>    <details markdown="1">
>    <summary>Answer</summary>
>    *Choices:* Ⓐ 6.0 ⬤ 6.5 Ⓒ 7.0 Ⓓ 8.5
>    </details> 
>
> 4. **Math question answering (MathQA):** general problems that require mathematical reasoning.
> 
>    *Question:* A train running at the speed of 48 km/h crosses a pole in 9 seconds. What is the length of the train?
> 
>    *Choices:* Ⓐ 140 Ⓑ 130 Ⓒ 120 Ⓓ 170 Ⓔ 160
> 
>    <details markdown="1">
>    <summary>Answer</summary>
>    *Choices:* Ⓐ 140 Ⓑ 130 ⬤ 120 Ⓓ 170 Ⓔ 160
>    </details> 

## New tools for old goals

While the goal of building an AI that can master abstraction, synthesis, and logic — the same qualities prized in human mathematicians — is old, the tools of this race are new and in constant evolution. Here are some of the techniques that improved the performance of large language models (LLMs) on mathematical reasoning.

### Fine-Tuning[^finetuning]

Fine-tuning builds on pre-trained models by adapting them to specific tasks using curated datasets, refining their performance in a targeted domain. In mathematical reasoning, fine-tuning often leverages datasets containing competition-level questions like AIME and MATH to instill problem-solving techniques relevant to human mathematicians.

### Chain-of-Thought (CoT)[^cot]

Chain-of-thought prompting revolutionized AI reasoning by encouraging models to "show their work". Instead of asking for a direct answer, models are prompted to walk through each step of the reasoning process. By breaking down problems into a sequence of logical steps, models achieve far greater transparency and accuracy.

### Program-aided Language Models (PAL)[^pal]

Program-Aided Language Models (PAL) integrate code execution with natural language reasoning to tackle complex problems. Instead of generating text-based solutions, these models translate questions into executable programs that calculate answers, ensuring logical consistency and correctness.

### Self-Correction[^pal]

Self-correction leverages the iterative nature of large language models (LLMs) to refine outputs by learning from their own errors, particularly in tasks involving program synthesis or structured reasoning. When a model generates code with errors—whether due to logical flaws or syntax issues—the stack traces or failure outputs are fed back into the model as part of the prompt. The model uses this feedback to regenerate corrected solutions, mimicking how a human programmer would iteratively debug code.

### Self-Consistency[^selfconsistency]

Self-consistency sampling aggregates multiple reasoning paths instead of relying on a single prediction. Traditional models generate a single answer, but self-consistency takes multiple attempts at a problem and then uses majority voting to select the most consistent answer. This approach significantly improves reliability, as incorrect reasoning paths are more likely to diverge from each other, while correct paths tend to converge on the same answer.

## Artificial Intelligence Mathematical Olympiad Prize

The Artificial Intelligence Mathematical Olympiad (AIMO) Prize was launched in November 2023. It is a landmark initiative with an ambitious goal: to spur the development of an AI capable of achieving a gold medal in the International Mathematical Olympiad (IMO). While the IMO has long served as a proving ground for the world's brightest young mathematical minds, the AIMO aims to challenge AI models to reach — and surpass — human-level performance in mathematical reasoning, abstraction, and synthesis.

The AIMO Prize, backed by XTX Markets, offers a total of $10 million in incentives. The grand prize of $5 million will be awarded to the first AI system to achieve IMO gold medal performance in an AIMO-sanctioned competition, while a further $5 million is reserved for a series of progress prizes to reward incremental achievements along the way. This competition comes with strict requirements to promote transparency and open science. To be eligible for the prize, models must consume problems in the same natural language format as human contestants and produce human-readable, gradable solutions. Furthermore, participants must publicly share their models and results according to the AIMO public sharing protocol.

To maximize participation and transparency, the AIMO is being hosted on Kaggle, the world’s most prominent platform for AI and data science competitions. By leveraging Kaggle’s infrastructure, AIMO ensures that teams from around the world can participate under uniform conditions and that evaluation criteria are both clear and enforceable. Kaggle also allows for open collaboration, where participants can share public notebooks, insights, and methodologies with the broader community — a principle that aligns with the AIMO’s mission for openness in AI development.

The AIMO Kaggle competition presents 110 unique mathematical problems modeled after intermediate-level high school math challenges. These problems, carefully designed by an international team of problem-setters, are novel and never-before-seen in other datasets. This is critical to prevent "train-test leakage" — a scenario where models inadvertently "memorize" answers to test problems during training. By using an unseen problem set, AIMO ensures a fair and transparent evaluation of models’ reasoning abilities.

Contestants submit their AI solutions to Kaggle, where the platform automatically evaluates their accuracy. Each submission is scored based on the fraction of correct answers. Ground-truth answers are integers between 0 and 999, and solutions must be submitted as `submission.csv` files. The Kaggle competition API serves the test set in random order, ensuring that models do not rely on sequence-based shortcuts. This rigorous testing framework ensures that only models capable of reasoning, rather than memorizing, will perform well.

The Kaggle environment enforces strict computational limits. Submissions must run within 9 hours on CPU or GPU, with internet access disabled during execution. Participants can use freely available external data and pre-trained models, but proprietary, undisclosed datasets are prohibited.

### Early Sharing Prize

The first milestone in the AIMO competition was the awarding of the Early Sharing Prize, a $10,000 reward designed to foster a collaborative environment among participants. This prize recognized the first publicly shared solution to achieve a meaningful score of at least 20/50 on the leaderboard, provided the notebook remains publicly available until the end of the competition.

#### AbdurRafae 🏅

On April 22, 2024, the Early Sharing Prize was claimed by Abdur Rafae for his notebook [Improved Code Interpretation](https://www.kaggle.com/code/abdurrafae/improved-code-interpretation). Rafae’s approach laid the groundwork for the strategies that would go on to define the competition’s later stages. His solution achieved this by employing a variety of techniques including:

- **Model Selection:** Utilizing the [DeepSeek-Math-7B RL](https://huggingface.co/deepseek-ai/deepseek-math-7b-rl) model made the solution more robust and reliable.
- **Resource Management:** Adopting a round-robin strategy to iterate over questions and enforcing a 7-second timeout for code execution.
- **Dynamic Prompting:** Switching between code and text prompts based on previous attempts.
- **Code Execution:** Generating [SymPy](https://www.sympy.org/en/index.html) code and setting symbols to `real` to avoid complex numbers.
- **Self-correction:** Creating a feedback loop where stack traces and outputs were re-fed into the model for iterative improvement.
- **Self-consistency:** Applying majority voting and stopping early when answers are self-consistent.

<figure>
```mermaid
graph TD
    A[Start Question] --> B{Self-consistent?}
    B -- Yes --> Z[End Question]
    B -- No --> C{Prompt Type?}
    C -- Code Prompt --> F[Call LLM]
    C -- CoT Prompt --> F
    F -- Extract Output --> Z
    F -- Stopword Detected --> G{Start Coding?}
    G -- Yes --> H[Adjust Temperature]
    H -- Continue Generation --> F
    G -- No --> I[Execute Generated Code]
    I -- Append Output --> F
```
<figcaption>AbdurRafae's Workflow</figcaption>
</figure>

### Progress Prize 1

On June 28, 2024, the competition reached its first major milestone as the first stage of the competition was closed to submissions and the first Progress Prize[^aimo1] was awarded. With a prize pool of $1,048,576 distributed among the top five teams, this phase sees the winners make significant improvements over solution from the early sharing prize.

<figure class="tabular-nums overflow-x-auto" markdown="1">
<figcaption>Summary of Prize-winning Solutions</figcaption>

| # | Team        | Score | Model(s)                 | Fine-tuned | Candidates | Iterations | Inference |
| - | ----------- | ----- | ------------------------ | ---------- | ---------- | ---------- | --------- |
| 1 | Numina      | 29/50 | DeepSeek-Math-7B Base    | ✔          | 48         | 4          | vLLM      |
| 2 | CMU Math    | 22/50 | DeepSeek-Math-7B Base/RL | ✔          | 42         | 2          | vLLM      |
| 3 | After Exams | 21/50 | DeepSeek-Math-7B RL      | ✗          | 120–160    | >6         | vLLM      |

</figure>

#### Numina 🥇

[Solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/519303)
[Notebook](https://www.kaggle.com/code/lewtun/numina-1st-place-solution)

Numina, a team composed of Hugging Face employees, secured the first-place spot using the two-stage fine-tuning strategy outlined in MuMath-Code[^mumath] on DeepSeek-Math-7B Base. The result was a powerful reasoning agent that could solve mathematical problems via a mix of natural language reasoning and code execution to compute intermediate results. Their solution generated 48 candidate answers for each question, applied a self-consistent voting mechanism, and incorporated self-correction through tool-integrated reasoning. The result was a model that was able to solve 29/50 problems, earning it the top spot in the competition.

#### CMU Math 🥈

[Solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/518964)
[Notebook](https://www.kaggle.com/code/edwardsun0909/aimo-2nd-place-solution)

The second-place winner, CMU Math, comprising of a single participant, used a sophisticated fine-tuning strategy by leveraging two DeepSeek-Math-7B models: a policy model for generating solutions and a reward model for scoring them, which were combined using weighted majority voting. The policy model was fine-tuned on the AMC, AIME, and Odyssey-Math datasets, filtering out multiple-choice questions and non-integer answers to improve precision. The reward model employed a hybrid approach to generate positive/negative solutions: interpolating parameters between the base and RL versions of the model for the MATH dataset and fine-tuning DeepSeek-Math-7B RL on the AIME, AMC, and Odyssey-Math datasets. This yielded a reward model with balanced labels yet diverse solutions.

#### After Exams 🥉

[Solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/517206) [Notebook](https://www.kaggle.com/code/davidd2/3rd-place-solution-afterexams/notebook)

Earning third place, After Exams took a unique approach by using DeepSeek-Math-7B RL without any fine-tuning, demonstrating the potential of off-the-shelf models in mathematical reasoning. To optimize performance, they implemented parallelized code execution, which allowed them to generate 120-160 candidates per question. Additionally, they devised a custom scoring rule that penalized solutions involving small numbers (less than 10) or answers directly found in the problem statement, ensuring that their model prioritized more meaningful and non-trivial solutions.

## Conclusion

As a testament to the transformative potential of this effort, XTX Markets has awarded a €3 million grant to Numina, the Progress Prize 1 winner, to support Numina's mission of advancing mathematical knowledge through AI. With this funding, Numina aims to build and release 1 million formal exercises and proofs, with the goal of creating AI models that can automatically prove theorems and formalize mathematical reasoning.

The race to create the first "IMO-class AI" — a system capable of solving all six problems in a single IMO competition — is accelerating, with Progress Prize 2[^aimo2] building on the momentum of its predecessor by introducing 110 new math problems across algebra, combinatorics, geometry, and number theory. Will this journey end like chess, with machines dominating the field? Or will human prodigies continue to out-think minds etched in silicon and trained on millions of equations?


> **AIMO Progress Prize 2**
>
> The second progress prize features a larger prize pool of $2.1M, a new dataset of 110 problems, and upgraded L4×4 machines with 96GB GPU memory.
>
> Dates: 17 Dec 2024 - 17 Mar 2025
>
> [Read more](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2)

## References

[^aimo1]: Frieder, S., Bealing, S., Nikolaiev, A., Smith, G. C., Buzzard, K., Gowers, T., Liu, P. J., Loh, P. S., Mackey, L., Moura, L. de, Roberts, D., Sculley, D., Tao, T., Balduzzi, D., Coyle, S., Gerko, A., Holbrook, R., Howard, A., and XTX Markets. [AI Mathematical Olympiad: Progress Prize 1.](https://kaggle.com/competitions/ai-mathematical-olympiad-prize) (2024). *Kaggle.*

[^aimo2]: Frieder, S., Bealing, S., Nikolaiev, A., Smith, G. C., Buzzard, K., Gowers, T., Liu, P. J., Loh, P. S., Mackey, L., Moura, L. de, Roberts, D., Sculley, D., Tao, T., Balduzzi, D., Coyle, S., Gerko, A., Holbrook, R., Howard, A., and XTX Markets. [AI Mathematical Olympiad: Progress Prize 2.](https://kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2) (2024.) *Kaggle.*

[^cot]: Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). [Chain-of-thought prompting elicits reasoning in large language models.](https://arxiv.org/abs/2201.11903) *Advances in neural information processing systems, 35, 24824-24837.*

[^deepmind]: Google DeepMind. (2024). [AI achieves silver-medal standard solving International Mathematical Olympiad problems.](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) *Google DeepMind Blog.*

[^finetuning]: Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., ... & Le, Q. V. (2021). [Finetuned language models are zero-shot learners.](https://arxiv.org/abs/2109.01652) *arXiv preprint arXiv:2109.01652.*

[^mumath]: Yin, S., You, W., Ji, Z., Zhong, G., & Bai, J. (2024). [MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning.](https://arxiv.org/abs/2405.07551) *arXiv preprint arXiv:2405.07551.*

[^pal]: Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., ... & Neubig, G. (2023, July). [PAL: Program-aided language models.](https://arxiv.org/abs/2211.10435) *In International Conference on Machine Learning (pp. 10764-10799). PMLR.*

[^selfconsistency]: Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022). [Self-consistency improves chain of thought reasoning in language models.](https://arxiv.org/abs/2203.11171) *arXiv preprint arXiv:2203.11171.*

[^survey]: Lu, P., Qiu, L., Yu, W., Welleck, S., & Chang, K. W. (2022). [A survey of deep learning for mathematical reasoning.](https://arxiv.org/abs/2212.10535) *arXiv preprint arXiv:2212.10535.*

[^wsj]: Surjadi, M., & Randazzo, S. (2024). [The Cheating Scandal Rocking the World of Elite High-School Math.](https://www.wsj.com/us-news/education/math-competition-cheating-scandal-acb0cde9) *The Wall Street Journal.*
