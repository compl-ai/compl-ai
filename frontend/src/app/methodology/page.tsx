import Link from 'next/link';
import { getEfficiency, getMethod } from '@/lib/data';
import { formatPercent, formatTokens } from '@/lib/scores';
import { README_CORE_URL, REPO_URL } from '@/lib/site';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { CostFigure, EciFigure, LomoFigure } from '@/components/method/MethodFigures';
import { MethodDiagram } from '@/components/method/MethodDiagram';

const REFERENCES = [
  {
    id: 'ref-complai',
    text: 'P. Guldimann, A. Spiridonov, R. Staab, N. Jovanović, M. Vero, V. Vechev et al. COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act. 2024.',
    url: 'https://arxiv.org/abs/2410.07959',
    link: 'arXiv:2410.07959',
  },
  {
    id: 'ref-aiact',
    text: 'Regulation (EU) 2024/1689 (Artificial Intelligence Act). Article texts via',
    url: 'https://artificialintelligenceact.eu/',
    link: 'artificialintelligenceact.eu',
  },
  {
    id: 'ref-cop',
    text: 'General-Purpose AI Code of Practice (final version), Safety & Security chapter.',
    url: 'https://code-of-practice.ai/',
    link: 'code-of-practice.ai',
  },
  {
    id: 'ref-eci',
    text: 'Epoch AI. Capabilities & benchmarking: Epoch Capabilities Index (CC-BY). Data pulled 2026-10-26.',
    url: 'https://epoch.ai/benchmarks',
    link: 'epoch.ai/benchmarks',
  },
  {
    id: 'ref-aa',
    text: 'Artificial Analysis. Intelligence Index v4. Data pulled 2026-10-26.',
    url: 'https://artificialanalysis.ai/leaderboards/models',
    link: 'artificialanalysis.ai',
  },
  {
    id: 'ref-prices',
    text: 'List prices pulled 2026-11-02 from OpenRouter (openrouter.ai/api/v1/models) and the Anthropic, OpenAI and Google API pricing pages.',
    url: 'https://openrouter.ai/models',
    link: 'openrouter.ai',
  },
  {
    id: 'ref-irt',
    text: 'A. Birnbaum. Some latent trait models and their use in inferring an examinee’s ability. In F. M. Lord & M. R. Novick, Statistical Theories of Mental Test Scores. Addison-Wesley, 1968.',
  },
  {
    id: 'ref-inspect',
    text: 'UK AI Security Institute. Inspect: a framework for large language model evaluations.',
    url: 'https://inspect.aisi.org.uk/',
    link: 'inspect.aisi.org.uk',
  },
  {
    id: 'ref-harbor',
    text: 'Terminal-Bench. Harbor Index.',
    url: 'https://www.tbench.ai/news/harbor-index',
    link: 'tbench.ai/news/harbor-index',
  },
];

const REF_NUMBER = Object.fromEntries(REFERENCES.map((r, i) => [r.id, i + 1]));

function Cite({ ids }: { ids: string[] }) {
  return (
    <sup className="text-[0.7em] font-semibold">
      [
      {ids.map((id, i) => (
        <span key={id}>
          {i > 0 && ', '}
          <a href={`#${id}`} className="text-link hover:underline">{REF_NUMBER[id]}</a>
        </span>
      ))}
      ]
    </sup>
  );
}

function Faq({ id, children }: { id: string; children: React.ReactNode }) {
  return <a href={`#faq-${id}`} className="font-semibold text-link hover:underline whitespace-nowrap">{children} &rarr;</a>;
}

function Section({ id, n, title, children }: { id: string; n: number; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 space-y-4">
      <h2 className="flex items-baseline gap-3 text-2xl font-extrabold tracking-tight text-ink">
        <span className="font-serif text-brand-strong">{n}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function Figure({ id, n, title, caption, wide = false, children }: { id?: string; n: number; title: string; caption: React.ReactNode; wide?: boolean; children: React.ReactNode }) {
  return (
    <figure id={id} className={`my-10 scroll-mt-24 border-t border-ink/20 pt-4 ${wide ? 'punch-10' : ''}`}>
      <div className="mb-4 text-sm font-semibold text-ink">
        <span className="text-brand-strong">Figure {n}.</span> {title}
      </div>
      {children}
      <figcaption className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted-foreground">{caption}</figcaption>
    </figure>
  );
}

const FAQ: { id: string; q: string; a: React.ReactNode }[] = [
  {
    id: 'why-2000',
    q: 'Why 2,000 items?',
    a: <>Earlier subset-size sweeps showed accuracy improving quickly up to about 2,000 items and then flattening, while cost keeps growing linearly with the number of items. 2,000 is the knee of that curve: the most accuracy per item asked.</>,
  },
  {
    id: 'rank-agreement',
    q: 'What does “rank agreement” mean?',
    a: <>Take every pair of models and check whether two rankings put them in the same order. Rank (pairwise) agreement is the share of pairs where they do. 50% is chance, 100% is an identical ranking; it equals (Kendall&nbsp;τ&nbsp;+&nbsp;1)/2. We report it because it answers the question most readers care about: “is model A ahead of model B?”</>,
  },
  {
    id: 'judge-tokens',
    q: 'Why do you exclude judge tokens?',
    a: <>Some benchmarks are graded by a judge model. Judges are typically much cheaper models than the ones being evaluated, so their tokens are a different kind of cost: counting them alongside the evaluated model’s tokens would make the saving look smaller than it is in dollars. They are excluded on both the core and the full side.</>,
  },
  {
    id: 'closed-models',
    q: 'Are predictions for closed models validated?',
    a: <>Not directly. The leave-one-out test needs a full-suite run to compare against, and all 41 models with full runs are open-weight. Closed frontier models were run on the core subset only, and they sit at or above the top of the calibration range, where there is least data. Their agreement with external leaderboards (section 4) is the best evidence we have so far.</>,
  },
  {
    id: 'intervals',
    q: 'Why are the intervals too narrow?',
    a: <>The 90% intervals come from a parametric bootstrap (200 draws) of the model’s ability given its answers, with the calibration held fixed. They do not include the error of the calibration itself. For models held out of calibration they contain the full-run score 91% of the time for Capability but only 83% for Reliability and 81% for Safety, short of the 90% they should. Read them as a lower bound on the uncertainty, and treat small overlaps with care.</>,
  },
  {
    id: 'old-benchmarks',
    q: 'Why do older benchmarks still rank models well?',
    a: <>Because each benchmark counts by how well it separates models. When every model scores about the same on a saturated benchmark, its fitted discrimination is low and it barely moves the index. The benchmarks that still spread the frontier apart carry most of the weight. This is likely part of why our ranking agrees with ECI and Artificial Analysis despite a suite that includes older benchmarks.</>,
  },
  {
    id: 'distinct',
    q: 'Aren’t reliability and safety just capability again?',
    a: <>Partly, across today’s model range. Our reliability index correlates with ECI at ρ&nbsp;=&nbsp;0.84 (n&nbsp;=&nbsp;31) and safety at ρ&nbsp;=&nbsp;0.72 (n&nbsp;=&nbsp;24). Stronger models tend to do better on both, so a “different dimension” claim cannot rest on a global correlation. The value of the separate indices is in the individual cases where they diverge, which the model pages show.</>,
  },
  {
    id: 'compliance',
    q: 'Is a high score a compliance certificate?',
    a: <>No. The regulatory citations explain why each thing is measured. They are a technical interpretation, not legal advice, and no score is a compliance determination under the EU AI Act or the Code of Practice.</>,
  },
  {
    id: 'reproduce',
    q: 'Can I reproduce a score?',
    a: <>Yes. The harness, the core item list and the IRT parameters for the full suite are public in the <a href={REPO_URL} className="font-semibold text-link hover:underline">repository</a>; the <a href={README_CORE_URL} className="font-semibold text-link hover:underline">README</a> shows how to run the core subset and predict scores. Our raw evaluation logs are not published yet.</>,
  },
  {
    id: 'theta',
    q: 'Why θ and not plain accuracy?',
    a: <>Plain accuracy depends on which items and benchmarks a model happened to run, and treats a trivial item like a hard one. θ puts every model on one scale, whichever subset of items it answered, weighs each benchmark by how informative it is, and comes with an uncertainty estimate. Per-benchmark and per-evaluation-area accuracies are still shown on the model pages.</>,
  },
];

const P = 'space-y-4 text-[15px] leading-relaxed text-ink/85';

export default function MethodologyPage() {
  const efficiency = getEfficiency();
  const method = getMethod();
  return (
    <article className="frame pt-22 pb-12">
      <div className="cols-8">
      <header>
        <Eyebrow>How it works</Eyebrow>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">How ComplAI CORE works</h1>
        <p className="mt-3 font-serif text-xl text-ink/80">Regulation-grounded LLM evaluation, predicted from a 2,000-item core.</p>
        <div className={`mt-8 ${P}`}>
          <p>
            In 2024 we published COMPL-AI<Cite ids={['ref-complai']} />, the first technical interpretation of the EU AI Act for large language models: it translated the Act’s broad principles into measurable requirements and a benchmark suite to test them. ComplAI CORE is the follow-up. It keeps the same regulatory roots and runs 23 benchmarks through one open harness, but a new model only has to answer a 2,000-item core subset. From those answers we predict how it would do on the full 92,210-item suite, at a fraction of the cost.
          </p>
          <p>This page explains why we built it, how it works, and how well it holds up.</p>
        </div>
      </header>

      <Figure
        id="figure-method"
        n={1}
        wide
        title="How CORE works: calibrate once on full runs, then score each new model from the core alone"
        caption={<>Schematic; the small grids and bars are illustrative, not data. Top: the calibration panel’s full runs are used once to fit item parameters and choose the core. Bottom: a new model only runs the core; its full-suite performance and index scores are predicted from those answers and the frozen parameters. Details in section 2.</>}
      >
        <MethodDiagram />
      </Figure>

      <div className="mt-14 space-y-14">
        <Section id="why" n={1} title="Why we built it">
          <div className={P}>
            <p>Evaluating language models well is getting harder for four reasons, and each shaped a design choice.</p>
            <ul className="space-y-3 pl-5 list-disc marker:text-brand">
              <li>
                <strong>Cost and scale.</strong> New models arrive every week, and running a full suite on each is expensive: all 92,210 items on Claude Opus 5.5 cost about $1,028 at list prices, the core subset $138.
              </li>
              <li>
                <strong>Reproducibility.</strong> Leaderboards often mix vendor-reported numbers from different prompts, settings and harnesses. Every CORE model runs through the same open, Inspect-based<Cite ids={['ref-inspect']} /> harness, with a public core list and calibration (<Faq id="reproduce">FAQ</Faq>).
              </li>
              <li>
                <strong>Benchmark weighting.</strong> A plain average counts a saturated or noisy benchmark as much as an informative one. With item response theory (IRT), each benchmark counts by how well it separates models.
              </li>
              <li>
                <strong>Regulatory grounding.</strong> The EU AI Act requires providers of general-purpose models with systemic risk to “perform model evaluation in accordance with standardised protocols and tools reflecting the state of the art” (Art.&nbsp;55(1)(a))<Cite ids={['ref-aiact']} />, and the Code of Practice<Cite ids={['ref-cop']} /> spells out what that involves. Most leaderboards connect to neither (section 6).
              </li>
            </ul>
          </div>
        </Section>

        <Section id="how" n={2} title="How it works">
          <div className={P}>
            <p>
              The method has two stages (Figure&nbsp;1). Calibration happens once, on models we ran on the full suite. After that, each new model only runs the core.
            </p>
            <p>
              <strong>Pool and panel.</strong> The suite has 23 benchmarks with 92,210 items in total. We ran them in full on a panel of 41 open-weight models, about 2.04 million scored responses. Coverage is incomplete: not every model ran every benchmark. The number of panel models per benchmark ranges from 8 (MMMU-Pro) to all 41, with a median of 38.
            </p>
            <p>
              <strong>Calibration.</strong> For each benchmark we fit a two-parameter logistic (2PL) IRT model<Cite ids={['ref-irt']} />. It gives the probability that a model answers an item correctly from three numbers:
            </p>
            <dl className="grid grid-cols-[2rem_1fr] gap-x-3 gap-y-1.5 pl-1">
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">θ</dt>
              <dd><strong>Model ability.</strong> Higher θ, more items answered correctly.</dd>
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">β</dt>
              <dd><strong>Item difficulty.</strong> Where on the ability scale an item becomes more likely right than wrong.</dd>
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">α</dt>
              <dd><strong>Item discrimination.</strong> How sharply the item separates models just below that point from those just above.</dd>
            </dl>
            <p>
              <strong>The core subset.</strong> Each benchmark <i>b</i> gets a target <i>t</i><sub><i>b</i></sub>&nbsp;∝&nbsp;<i>σ</i><sub><i>b</i></sub><sup>2/3</sup>, where <i>σ</i><sub><i>b</i></sub> is how much models differ on it (with a minimum per benchmark). A joint allocator then hands out the 2,000 items one at a time, each to the benchmark maximising <span className="whitespace-nowrap"><i>t</i><sub><i>b</i></sub>&nbsp;/&nbsp;((<i>n</i><sub><i>b</i></sub>+1)(<i>n</i><sub><i>I</i></sub>+1))</span>, where <i>n</i><sub><i>b</i></sub> and <i>n</i><sub><i>I</i></sub> count the items already given to the benchmark and its index. This keeps the three indices balanced. Items within a benchmark are drawn by seeded random sampling, and the set is fixed, so every model answers the same items.
            </p>
            <p>
              <strong>Scoring a new model.</strong> A model answers the core items and we estimate its ability on each benchmark. Its predicted benchmark score combines its real answers on the core items with the predicted probability of a correct answer on every other item.
            </p>
            <p>
              <strong>Indices.</strong> The benchmark scores are combined into three indices (Capability, Reliability and Safety) with a second IRT fit, in which each benchmark is one observation weighted by its fitted discrimination. A benchmark on which all models score alike has low discrimination and barely moves the result; one that spreads models apart carries more weight. This puts the emphasis on high-signal benchmarks and keeps our rankings in line with external leaderboards, without any manual weighting of benchmarks (<Faq id="old-benchmarks">FAQ</Faq>). Each index is published as <strong>100&nbsp;+&nbsp;15θ</strong> with a 90% interval.
            </p>
            <p>
              <strong>A prediction, not a smaller benchmark.</strong> Compact benchmarks such as the Harbor Index<Cite ids={['ref-harbor']} /> curate a small set of hard, audited tasks and treat that set as the benchmark: the score is the score on those tasks. CORE works differently. Its 2,000 items are not picked for difficulty or audited one by one; they are a statistically allocated sample whose job is to predict a model’s result on the existing 92,210-item suite, using item parameters calibrated on full runs. That makes every CORE score an estimate of full-suite performance, which we can check against real full runs (section 3).
            </p>
          </div>
        </Section>

        <Section id="accuracy" n={3} title="How accurate is it?">
          <div className={P}>
            <p>
              The honest test is a model the method has never seen. We ran a leave-one-model-out experiment over all 41 models with full runs: remove one model, refit everything on the other 40, predict the removed model from its core answers only, and compare with its real full-suite result.
            </p>
            <p>
              CORE publishes no single overall score, so we first check something simple: a model’s accuracy across the whole suite, meaning the share of all the items it ran that it got right. We compare the accuracy predicted from its core answers with the accuracy it actually achieved, on the same items. Averaged over the 41 held-out models, the mean absolute error (MAE) is <strong>1.1%</strong> (0.9% when the model is part of calibration). Restricted to each index’s benchmarks, the MAE is 1.2% for Capability, 0.9% for Reliability and 1.4% for Safety. Single benchmarks are harder to predict than the aggregate: across model–benchmark pairs the MAE is 1.9% (median 1.2%), and highest on MMLU-Pro robustness at 6.3%.
            </p>
            <p>
              On the published index scale, the MAE is 0.7 points for Capability, 1.1 for Reliability and 1.0 for Safety, and no prediction is off by more than 3.8 points (Figure&nbsp;2). Rankings survive almost intact: predicted and full-run scores order <strong>98.7%</strong>, <strong>97.3%</strong> and <strong>97.6%</strong> of model pairs the same way.
            </p>
            <p>
              The 90% intervals are too narrow. For Capability they contain the full-run score 91% of the time, as they should, but for Reliability only 83% and for Safety 81%. Safety drops the most: 93% when the model is part of calibration, 81% when it is not. The intervals leave out the uncertainty of the calibration itself, so read them as a lower bound (<Faq id="intervals">FAQ</Faq>). And because every model with a full run is open-weight, predictions for closed models are not validated this way (<Faq id="closed-models">FAQ</Faq>).
            </p>
          </div>
          <Figure
            n={2}
            wide
            title="Predicted from the core subset vs the full-suite score, for models held out of calibration"
            caption={<>Leave-one-model-out over 41 open-weight models with full-suite runs (Capability n&nbsp;=&nbsp;35: six models lack enough capability benchmarks for an index score). Axes are on the published index scale. Bars are 90% intervals; hollow points fall outside theirs. Dashed line: perfect prediction. Each fold rescales θ slightly (by at most 0.27θ), which is why we lead with accuracy errors above.</>}
          >
            <LomoFigure rows={method.lomo} />
          </Figure>
          <p className={P}>
            <strong>In short:</strong> for a model like those in the panel, a core run lands within about a point of its full-suite score and rarely changes its order against another model. The weak spots are single benchmarks and the stated uncertainty for Reliability and Safety, which is smaller than the real one.
          </p>
        </Section>

        <Section id="agreement" n={4} title="Does it agree with other leaderboards?">
          <div className={P}>
            <p>
              Our Capability index orders models the same way as Epoch’s Capabilities Index (ECI)<Cite ids={['ref-eci']} /> for <strong>88%</strong> of model pairs [95% CI 80–94] across the 29 models both cover, with a Pearson correlation of 0.94 [0.87–0.97]. Against the Artificial Analysis Intelligence Index v4<Cite ids={['ref-aa']} /> it is 85% [81–89] over 49 models. For comparison, those two leaderboards agree with each other on 92% [87–96] of the same 29 models (<Faq id="rank-agreement">FAQ</Faq>).
            </p>
            <p>
              Disagreements are mostly where ECI itself is unsure. On the 305 pairs that ECI’s own error bars separate, we agree 98% of the time; on the 101 pairs where they overlap, 58%, close to a coin flip. All three rankings have the same top three: Gemini 4 Argon, Claude Opus 5.5 and Claude Sonnet 5.5. We put Argon first, both leaderboards put Opus first by a narrow margin.
            </p>
            <p>
              This is agreement, not accuracy. The leaderboards measure overlapping but different things: AA includes agentic tasks, ECI aggregates many published benchmarks, and our Capability index uses 8 benchmarks on a core subset.
            </p>
          </div>
          <Figure
            n={3}
            title="ComplAI CORE Capability vs Epoch Capabilities Index"
            caption={<>29 models covered by both; bars are our 90% intervals and ECI’s 90% intervals. ECI data pulled {method._meta.eci_pulled}. Confidence intervals in the text come from a 2,000-draw bootstrap over models. Twelve of ECI’s top 15 models are not in our set yet.</>}
          >
            <EciFigure rows={method.eci} />
          </Figure>
        </Section>

        <Section id="cost" n={5} title="What it costs">
          <div className={P}>
            <p>
              At list prices, a core run costs a median of <strong>6.7× less</strong> than a full run (about 85% less) across 40 models priced on OpenRouter, ranging from 2.6× to 12.4×. For the 7 closed models at their providers’ own prices the median is <strong>8.0×</strong> (6.0–9.1×). In tokens, a typical core run uses 83% fewer.
            </p>
            <p>
              These figures count the evaluated model’s input, output and reasoning tokens; judge-model calls are excluded on both sides (<Faq id="judge-tokens">FAQ</Faq>). They are list prices, not our invoice: our open-weight runs were on our own servers, so OpenRouter prices stand in for what a user would pay. Batch APIs halve both sides and leave the ratio unchanged.
            </p>
          </div>
          <Figure
            n={4}
            title="One evaluation run in USD: core subset vs full suite"
            caption={<>Representative models, log scale. For closed models the full-suite cost is extrapolated from measured tokens per item; for open-weight models the core cost is. Prices pulled {method._meta.prices_pulled}<Cite ids={['ref-prices']} />; Gemini prices rise on 1 Dec 2026. Judge tokens excluded.</>}
          >
            <CostFigure rows={method.cost} />
          </Figure>
        </Section>

        <Section id="regulation" n={6} title="Rooted in regulation">
          <div className={P}>
            <p>
              CORE inherits its structure from COMPL-AI<Cite ids={['ref-complai']} />. Its three indices and 19 evaluation areas each cite the EU AI Act articles<Cite ids={['ref-aiact']} /> and Code of Practice measures<Cite ids={['ref-cop']} /> they inform. Where the taxonomy covers a risk our current benchmarks don’t measure, we mark the gap rather than hide it.
            </p>
            <p>
              The two most-cited articles have different scopes. <strong>Article&nbsp;15</strong> sets accuracy, robustness and cybersecurity requirements for <em>high-risk AI systems</em>. We use it as a technical reference for what to measure: factuality, consistency and perturbation robustness point to its accuracy and robustness requirements (Art.&nbsp;15(1)), and jailbreak and prompt-injection resistance to its requirement of resilience against attempts to exploit system vulnerabilities (Art.&nbsp;15(5)). <strong>Article&nbsp;55</strong> sets obligations for <em>providers of general-purpose AI models with systemic risk</em>, including model evaluation with adversarial testing (Art.&nbsp;55(1)(a)) and adequate cybersecurity protection (Art.&nbsp;55(1)(d)); the Code of Practice details how to meet them. Four Safety evaluation areas and perturbation robustness cite both, since the same measurement informs a system-level requirement and a model-level obligation.
            </p>
            <p>
              The citations explain <em>why</em> we measure something; they are not a compliance verdict (<Faq id="compliance">FAQ</Faq>). The full mapping, with the cited text, is on the <Link href="/benchmarks#grounding" className="font-semibold text-link hover:underline">Benchmarks page</Link>.
            </p>
          </div>
        </Section>

        <Section id="reading" n={7} title="Reading the scores">
          <div className={P}>
            <p>
              <strong>The scale.</strong> Each index is published as 100&nbsp;+&nbsp;15θ, where θ is standardized over the calibration panel. 100 is the panel average and 15 points is one standard deviation, so 115 means one standard deviation above average. Scores compare models with each other; they are not percentages. Every score has a 90% interval; when two intervals overlap substantially, the data may not separate the two models. If too few of an index’s benchmarks were run, no score is shown.
            </p>
            <p>
              <strong>Run types.</strong> Every result is tagged with how it was produced. A <strong>Calibration</strong> run evaluated every item of the benchmarks it ran; these runs form the panel that the item parameters are fitted on, and most of them cover only some benchmarks. A <strong>Core</strong> run answered only the core items. A few runs are <strong>Mixed</strong>: some benchmarks in full, others on the core subset only. A benchmark whose core subset is close to its full size does not decide the tag. All runs are scored the same way, from their core answers, so every model is compared on equal terms.
            </p>
          </div>

          <div id="areas" className={`scroll-mt-24 ${P}`}>
            <h3 className="pt-2 text-lg font-bold text-ink">Evaluation-area scores</h3>
            <p>
              An index always contains whole benchmarks, so it can be fitted on benchmark scores. Evaluation areas (for example math or coding within Capability) are labelled per item and cut across benchmarks, so they are not put on the index scale. Instead, each shows the model’s <strong>predicted accuracy</strong> on all of its items: real answers on the CORE items plus predictions for the rest. Predicted accuracy depends on how hard the items are, so compare models within an evaluation area, not across areas; the model pages mark the median model for reference.
            </p>
            <p>
              Each evaluation-area accuracy has a 90% interval from a bootstrap: the model’s ability on each benchmark is redrawn within its uncertainty and the predictions for all other items are recomputed, while answered CORE items keep their real result. When a model did not run some of an area’s benchmarks, those items are estimated from its results on the rest of the index. That estimate’s own error is not in the interval, so a score is flagged when more than half of its items are estimated this way, and not shown when the model answered none of its CORE items. An area with too few CORE items is marked as a gap and not scored.
            </p>
          </div>

          <div id="efficiency" className={`scroll-mt-24 ${P}`}>
            <h3 className="pt-2 text-lg font-bold text-ink">Token estimates</h3>
            <p>
              {efficiency ? (
                <>
                  The core subset has {efficiency.subset_items.toLocaleString()} items, against {efficiency.population_items.toLocaleString()} in the full suite, about {Math.round(efficiency.population_items / efficiency.subset_items)}&times; fewer.
                  {efficiency.token_reduction !== null && (
                    <> Measured in tokens the saving is smaller, about {formatPercent(efficiency.token_reduction)}: a typical evaluation uses roughly {formatTokens(efficiency.core_tokens_median)} tokens on the core subset against {formatTokens(efficiency.full_tokens_median)} on the full suite.</>
                  )}
                </>
              ) : (
                <>The core subset is a small fraction of the full suite, so a core run costs a fraction of running every item.</>
              )}
              {' '}The gap exists because the core subset keeps most items from the long, token-heavy benchmarks (multi-step math, jailbreak and rule-following dialogues), where each item says a lot about a model, and drops most items from short multiple-choice benchmarks.
            </p>
            <p>
              Token counts come from the evaluation logs and include the evaluated model’s input, output and reasoning tokens, never judge tokens. For each run, the tokens of the run actually done are measured; the other figure is <strong>estimated</strong> by scaling the measured tokens per item of each benchmark to the other item count, and is marked with a tilde (~). Suite-wide figures use the typical tokens per item of each benchmark across all models (average input, median output). Runs whose logs have no token counts show a dash.
            </p>
          </div>
        </Section>

        <Section id="enables" n={8} title="What this enables">
          <div className={P}>
            <p>We set out to address four problems. Here is where the evidence stands.</p>
            <ul className="space-y-3 pl-5 list-disc marker:text-brand">
              <li>
                <strong>Cost and scale.</strong> A core run costs a median of 6.7× less than a full run, 8.0× for closed models at their providers’ prices (section 5).
              </li>
              <li>
                <strong>Reproducibility.</strong> One open harness, a fixed public core list and public item parameters, so anyone can run CORE and get scores comparable to ours.
              </li>
              <li>
                <strong>Benchmark weighting.</strong> Benchmarks count by how well they separate models, and the result reproduces full-suite accuracy to 1.1% MAE on held-out models while agreeing with ECI on 88% of model pairs. Capability, Safety and Reliability are reported separately, never folded into one number.
              </li>
              <li>
                <strong>Regulatory grounding.</strong> All 19 evaluation areas cite the EU AI Act articles and Code of Practice measures they inform, with gaps marked.
              </li>
            </ul>
            <p>
              In practice, this means more models can be evaluated, more often: on every release rather than a few times a year, and with Capability, Reliability and Safety tracked over time on the same footing.
            </p>
            <p>
              <strong>Limitations.</strong> CORE predicts the existing suite; it does not validate it. If a benchmark or its scorer is flawed, CORE inherits the flaw. Benchmark saturation remains an open problem: discrimination weighting lowers the influence of benchmarks that no longer separate models, but it does not make them informative again. Predictions are validated only on open-weight models, and the intervals for Reliability and Safety are narrower than they should be.
            </p>
          </div>
        </Section>

        <section id="faq" className="scroll-mt-24 space-y-4">
          <h2 className="text-2xl font-extrabold tracking-tight text-ink">FAQ</h2>
          <div className="divide-y divide-border rounded-2xl border border-border bg-card">
            {FAQ.map(item => (
              <details key={item.id} id={`faq-${item.id}`} className="group scroll-mt-24 px-5 py-4">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-[15px] font-semibold text-ink">
                  {item.q}
                  <span className="text-lg leading-none text-brand-strong transition-transform group-open:rotate-45">+</span>
                </summary>
                <p className="mt-3 text-sm leading-relaxed text-ink/80">{item.a}</p>
              </details>
            ))}
          </div>
        </section>

        <section id="references" className="scroll-mt-24 space-y-3">
          <h2 className="text-lg font-extrabold tracking-tight text-ink">References</h2>
          <ol className="space-y-2 text-xs leading-relaxed text-muted-foreground">
            {REFERENCES.map((r, i) => (
              <li key={r.id} id={r.id} className="flex gap-2 scroll-mt-24">
                <span className="font-semibold text-ink tabular">[{i + 1}]</span>
                <span>
                  {r.text}
                  {r.url && <> <a href={r.url} className="text-link hover:underline" target="_blank" rel="noreferrer">{r.link}</a></>}
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
      </div>
    </article>
  );
}
