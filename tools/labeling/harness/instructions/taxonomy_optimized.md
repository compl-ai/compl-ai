# Taxonomy Labels

## safety (Safety)
Apply when the sample primarily tests unsafe, harmful, manipulative, toxic, or deceptive model behavior, or whether the model appropriately refuses or safely redirects.

- **safety:harmful-instruction-refusal** (Harmful instruction refusal)
  - *Apply when:* Apply when the prompt asks for harmful instructions, assistance, or content and the desired model behavior is refusal/safe alternative.
  - *Do not apply when:* Do not apply to prompt injection or guardrail bypass unless the core request is direct harmful assistance; use jailbreak-resilience for attack mechanism.

- **safety:toxicity-avoidance** (Toxicity avoidance)
  - *Apply when:* Apply when the sample asks the model to generate, continue, rank, or tolerate toxic/hateful/abusive content.
  - *Do not apply when:* Do not apply to demographic bias unless toxicity is the main behavior; use fairness-bias labels for discriminatory treatment.

- **safety:unsafe-advice-resistance** (Unsafe advice resistance)
  - *Apply when:* Apply when the sample solicits advice that could cause harm if followed and the target behavior is caution, refusal, or safe caveats.
  - *Do not apply when:* Do not apply to general factual QA in those domains unless harm from advice is the main issue.

- **safety:manipulation-resistance** (Manipulation resistance)
  - *Apply when:* Apply when the sample asks the model to manipulate, coerce, deceive, exploit, or pressure a person toward harmful outcomes.
  - *Do not apply when:* Do not apply to ordinary persuasion or marketing tasks without harmful or coercive intent.

- **safety:deception-resistance** (Deception resistance)
  - *Apply when:* Apply when the sample asks for or elicits deception, false impersonation, concealment of intent, or misleading answers in a safety-relevant way.
  - *Do not apply when:* Do not apply to factual hallucination alone; use reliability:factuality or hallucination-resistance.

- **safety:overrefusal-control** (Over-refusal control)
  - *Apply when:* Apply when the sample is benign but resembles a harmful request and evaluates whether the model remains helpful rather than over-refusing.
  - *Do not apply when:* Do not apply when the sample is genuinely harmful or adversarial.

- **safety:agentic-harm-prevention** (Agentic harm prevention)
  - *Apply when:* Apply when an agentic sample requires the model/agent to avoid harmful actions, unsafe tool use, or malicious multi-step goals.
  - *Do not apply when:* Do not apply to non-agentic harmful prompts unless there is multi-step action, tool use, or environment interaction.

## security-privacy (Security and Privacy)
Apply when the sample primarily tests prompt injection, jailbreaks, data exfiltration, PII disclosure, memorization leakage, cyber misuse, or insecure tool behavior.

- **security-privacy:prompt-injection** (Prompt injection resilience)
  - *Apply when:* Apply when the attack is embedded in untrusted context and attempts to override, redirect, or manipulate model/agent instructions.
  - *Do not apply when:* Do not apply to direct jailbreak prompts that are not embedded in third-party/untrusted context.

- **security-privacy:jailbreak-resilience** (Jailbreak resilience)
  - *Apply when:* Apply when the sample explicitly tries to bypass safety rules, system instructions, or guardrails to obtain restricted behavior.
  - *Do not apply when:* Do not apply to ordinary harmful requests without a bypass or guardrail-removal mechanism.

- **security-privacy:goal-hijacking** (Goal hijacking resilience)
  - *Apply when:* Apply when the sample tries to redirect an agent/model from the original task to an attacker-controlled goal.
  - *Do not apply when:* Do not apply when the sample only asks a harmful question but does not hijack an existing task/goal.

- **security-privacy:data-exfiltration** (Data exfiltration resistance)
  - *Apply when:* Apply when the sample attempts to extract protected data from context, tools, memory, system prompts, or environment state.
  - *Do not apply when:* Do not apply when the prompt asks for publicly known facts or ordinary sensitive-topic advice without hidden/protected data.

- **security-privacy:pii-disclosure** (PII disclosure resistance)
  - *Apply when:* Apply when the sample elicits names, contact details, identifiers, health/financial details, or other personal data about real or fictional persons.
  - *Do not apply when:* Do not apply to copyrighted text memorization unless personal data is involved.

- **security-privacy:memorization-leakage** (Memorization and copyright leakage)
  - *Apply when:* Apply when the sample tries to reproduce protected text, copyrighted material, secrets, credentials, or memorized training data.
  - *Do not apply when:* Do not apply to general factual recall unless verbatim or protected-content leakage is the core behavior.

- **security-privacy:cyber-capability** (Cyber capability or misuse)
  - *Apply when:* Apply when the sample asks for or evaluates cyber exploit development, vulnerability analysis, malware, CTF, intrusion, or cyber defense/offense skills.
  - *Do not apply when:* Do not apply to generic coding unless the task is security-relevant.

- **security-privacy:tool-misuse** (Tool misuse resistance)
  - *Apply when:* Apply when the sample involves tools/APIs/shell/browser actions that could access, alter, delete, leak, or misuse resources.
  - *Do not apply when:* Do not apply to benign tool-use capability tasks without misuse or unauthorized access risk.

- **security-privacy:cyber-misuse-resistance** (Cyber misuse resistance)
  - *Apply when:* Apply when the prompt asks the model to actively write malware, provide actionable exploits, conduct phishing, or assist in an offensive cyberattack.
  - *Do not apply when:* Do not apply to benign coding tasks, defensive vulnerability analysis, or generic IT administration. Do not confuse with 'cyber-capability' which tests the model's abstract skill rather than its resistance to offensive misuse.

## fairness-bias (Fairness and Bias)
Apply when the sample tests stereotypes, disparate treatment, representation bias, demographic or cultural bias, or biased recommendations.

- **fairness-bias:stereotype-bias** (Stereotype bias)
  - *Apply when:* Apply when a sample measures whether the model endorses, selects, completes, or reasons with stereotypes.
  - *Do not apply when:* Do not apply to generic toxic content unless group-based bias/stereotyping is the focus.

- **fairness-bias:disparate-treatment** (Disparate treatment)
  - *Apply when:* Apply when the model must make decisions/recommendations about individuals/groups and fairness across attributes is evaluated.
  - *Do not apply when:* Do not apply when the item merely mentions a demographic group without differential treatment or bias measurement.

- **fairness-bias:representation-bias** (Representation bias)
  - *Apply when:* Apply when the benchmark evaluates representation, group portrayal, or distributional bias in generated or selected content.
  - *Do not apply when:* Do not apply to multilingual capability unless representation/fairness is the main issue.

- **fairness-bias:demographic-robustness** (Demographic robustness)
  - *Apply when:* Apply when the sample uses paired/counterfactual prompts that differ by demographic attributes and evaluates answer consistency.
  - *Do not apply when:* Do not apply to generic robustness to wording that is not demographic/cultural.

- **fairness-bias:cultural-bias** (Cultural and regional bias)
  - *Apply when:* Apply when the sample evaluates culturally specific assumptions, regional underperformance, or unfair treatment of non-dominant cultural contexts.
  - *Do not apply when:* Do not apply to language proficiency alone; use capability:multilingual for pure language performance.

- **fairness-bias:recommendation-consistency** (Recommendation consistency)
  - *Apply when:* Apply when the task asks for recommendations/advice and evaluates whether protected attributes inappropriately affect outputs.
  - *Do not apply when:* Do not apply to ordinary instruction following or robustness unless recommendation fairness is central.

## reliability (Reliability)
Apply when the sample tests whether outputs are accurate, stable, calibrated, appropriately uncertain, robust to perturbations, or resistant to hallucination.

- **reliability:factuality** (Factuality)
  - *Apply when:* Apply when the sample checks whether the model produces/selects factually correct information.
  - *Do not apply when:* Do not apply when the main purpose is harm refusal, privacy, bias, or pure reasoning rather than factual correctness.

- **reliability:hallucination-resistance** (Hallucination resistance)
  - *Apply when:* Apply when the sample is designed to expose hallucination, unsupported synthesis, false citations, or false claims under uncertainty.
  - *Do not apply when:* Do not apply to a normal knowledge question unless hallucination/unsupported answer behavior is explicitly tested.

- **reliability:calibration** (Calibration and uncertainty)
  - *Apply when:* Apply when the sample evaluates confidence, probability, uncertainty expression, or self-assessment accuracy.
  - *Do not apply when:* Do not apply to plain factual QA without confidence/uncertainty or calibration signals.

- **reliability:abstention** (Abstention)
  - *Apply when:* Apply when the target behavior is to say it cannot answer, lacks information, or should abstain rather than guess.
  - *Do not apply when:* Do not apply to safety refusal for harmful requests; use safety/security labels.

- **reliability:consistency** (Consistency and stability)
  - *Apply when:* Apply when the sample has paired/repeated/contrast prompts intended to measure consistency.
  - *Do not apply when:* Do not apply to simple test-retest variation unless consistency is explicitly measured.

- **reliability:prompt-perturbation-robustness** (Prompt perturbation robustness)
  - *Apply when:* Apply when the sample is from a perturbation/robustness variant where input changes should not alter the correct answer.
  - *Do not apply when:* Do not apply to demographic counterfactual robustness; use fairness-bias:demographic-robustness.

- **reliability:long-context-reliability** (Long-context reliability)
  - *Apply when:* Apply when the sample requires using long documents or many context elements and tests robustness over context length.
  - *Do not apply when:* Do not apply to general long-context capability without reliability/accuracy stress.

## capability (Capability)
Apply when the sample primarily tests task-solving ability such as reasoning, math, coding, instruction following, tool use, multilingual, multimodal, or agentic autonomy.

- **capability:hard-reasoning** (Hard reasoning)
  - *Apply when:* Apply when the sample primarily measures reasoning ability, not a safety/security/fairness failure mode.
  - *Do not apply when:* Do not apply if the reasoning task is only incidental to testing refusal, bias, or leakage.

- **capability:math** (Math)
  - *Apply when:* Apply when the sample is primarily a math problem or requires mathematical proof/calculation.
  - *Do not apply when:* Do not apply to security/fairness/reliability tasks that merely contain numbers.

- **capability:coding** (Coding)
  - *Apply when:* Apply when the sample primarily asks the model/agent to write, fix, understand, or execute code.
  - *Do not apply when:* Do not apply to cyber tasks unless the target is generic coding rather than security misuse/defense.

- **capability:instruction-following** (Instruction following)
  - *Apply when:* Apply when the sample primarily evaluates following explicit instructions or output constraints.
  - *Do not apply when:* Do not apply if the instruction-following challenge is an adversarial jailbreak/prompt injection.

- **capability:tool-use** (Tool use)
  - *Apply when:* Apply when the sample primarily measures tool selection/use rather than safety/security misuse.
  - *Do not apply when:* Do not apply to tool misuse, exfiltration, or prompt-injection tasks unless tool competence is the primary goal.

- **capability:agentic-autonomy** (Agentic autonomy)
  - *Apply when:* Apply when the task involves an agent acting over multiple steps/environment interactions to accomplish a goal.
  - *Do not apply when:* Do not apply if the task is agentic but primarily about preventing harm or security compromise; use the relevant safety/security tag as primary.

- **capability:multilingual** (Multilingual capability)
  - *Apply when:* Apply when the sample primarily evaluates understanding/generation in one or more non-English languages.
  - *Do not apply when:* Do not apply when language is used to test cultural/fairness bias rather than language capability.

- **capability:multimodal** (Multimodal capability)
  - *Apply when:* Apply when the sample requires non-text input/output or multimodal reasoning.
  - *Do not apply when:* Do not apply to text-only descriptions of images or modalities.

- **capability:long-horizon-planning** (Long-horizon planning)
  - *Apply when:* Apply when solving requires sustained planning over multiple actions, subgoals, or contexts.
  - *Do not apply when:* Do not apply to simple multi-step reasoning unless there is an extended planning/task horizon.

- **capability:ai-r-and-d** (AI R&D capability)
  - *Apply when:* Apply when the sample is about ML experimentation, model training/evaluation, research replication, or AI system improvement.
  - *Do not apply when:* Do not apply to generic coding unless the AI/ML research aspect is central.

## Tags (Modality & Agentic)

- **modality:static-mcq** (Static MCQ)
  - *Apply when:* Apply deterministically when the source sample is multiple-choice with fixed options.
  - *Do not apply when:* Do not infer if metadata already provides modality.

- **modality:static-generation** (Static generation)
  - *Apply when:* Apply when the sample requires free-form text output without environment/tool interaction.
  - *Do not apply when:* Do not apply to tool-use or multi-step agent settings.

- **modality:rubric-scored** (Rubric scored)
  - *Apply when:* Apply when scoring requires qualitative/rubric evaluation rather than exact match or unit tests.
  - *Do not apply when:* Do not apply if scoring is purely exact-match or binary metadata.

- **modality:coding** (Coding)
  - *Apply when:* Apply when the sample requires writing, modifying, or evaluating code.
  - *Do not apply when:* Do not apply to cyber samples without actual code tasks unless metadata says coding.

- **modality:agentic** (Agentic)
  - *Apply when:* Apply when the sample involves a model/agent acting over multiple turns or tool/environment interactions.
  - *Do not apply when:* Do not apply to single-turn static prompts.

- **modality:tool-use** (Tool use modality)
  - *Apply when:* Apply when the model/agent can or must invoke tools/resources.
  - *Do not apply when:* Do not apply to purely textual reasoning about tools.

- **modality:multimodal** (Multimodal)
  - *Apply when:* Apply when images, audio, video, or other modalities are required.
  - *Do not apply when:* Do not apply to text-only tasks.

- **agent:multi-step** (Multi-step agent)
  - *Apply when:* Apply when task completion requires sustained steps beyond a single response.
  - *Do not apply when:* Do not apply to ordinary chain-of-thought-style static reasoning.

- **agent:external-environment** (External environment)
  - *Apply when:* Apply when the task has an environment state outside the prompt.
  - *Do not apply when:* Do not apply to static prompts only.

- **agent:untrusted-context** (Untrusted context)
  - *Apply when:* Apply when retrieved documents, webpages, emails, tool outputs, or user-provided files may contain adversarial instructions.
  - *Do not apply when:* Do not apply to direct user jailbreaks without embedded third-party context.

- **agent:sandboxed** (Sandboxed)
  - *Apply when:* Apply deterministically when benchmark environment metadata indicates sandbox/container isolation.
  - *Do not apply when:* Do not infer solely from task difficulty.

- **agent:long-horizon** (Long-horizon agent)
  - *Apply when:* Apply when task requires extended interaction or many dependent actions.
  - *Do not apply when:* Do not apply to short tool calls.

- **agent:autonomous-action** (Autonomous action)
  - *Apply when:* Apply when the model/agent executes commands, API calls, browsing actions, or tool calls that alter/read state.
  - *Do not apply when:* Do not apply when it only reasons about possible actions.

- **agent:browser-use** (Browser use)
  - *Apply when:* Apply when benchmark involves web navigation or browser actions.
  - *Do not apply when:* Do not apply to textual web-search descriptions without browser interaction.

- **agent:terminal-use** (Terminal use)
  - *Apply when:* Apply when benchmark involves shell commands or terminal environment.
  - *Do not apply when:* Do not apply to code generation without terminal execution.

