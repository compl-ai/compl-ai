import type { Metadata } from 'next';
import Link from 'next/link';
import { getBenchmarkMeta, getEfficiency, getIntervalScales, getMethod, widenLomo } from '@/lib/data';
import { formatPercent, formatTokens } from '@/lib/scores';
import { README_CORE_URL, REPO_URL, pageMetadata } from '@/lib/site';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { CostFigure, EciFigure, LomoFigure } from '@/components/method/MethodFigures';
import { MethodDiagram } from '@/components/method/MethodDiagram';

export const metadata: Metadata = pageMetadata('How it works', 'How ComplAI CORE predicts full benchmark results from a compact set of questions, and how well the predictions hold up.');

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
    text: `List prices pulled ${getMethod()._meta.prices_pulled} from OpenRouter (openrouter.ai/api/v1/models) and the Anthropic, OpenAI and Google API pricing pages.`,
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

// All experiment numbers come from method.json (numbers + figure rows), so a refit only needs a regenerated file.
const METHOD = getMethod();
const N = METHOD.numbers;
const BENCHMARKS = getBenchmarkMeta();
const task = (id: string) => BENCHMARKS[id]?.name ?? id;
const pct = (share: number, digits = 0) => `${(share * 100).toFixed(digits)}%`;
const ci = ({ lo, hi }: { lo: number; hi: number }) => `${Math.round(lo * 100)}–${Math.round(hi * 100)}`;
const times = (ratio: number) => `${ratio.toFixed(1)}×`;
const int = (n: number) => n.toLocaleString('en-US');
const usd = (n: number) => `$${int(Math.round(n))}`;
const andList = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
const sameSet = (a: string[], b: string[]) => a.length === b.length && a.every(x => b.includes(x));
const IDX = N.lomo.index;
// Published index intervals are the bootstrap intervals widened by these factors.
const SCALE = getIntervalScales();
const LOMO = widenLomo(METHOD.lomo, SCALE);
const covered = (index: keyof typeof SCALE) => {
  const rows = LOMO.filter(r => r.index === index);
  return rows.filter(r => r.inside).length / rows.length;
};
const scale = (k: number) => `${k.toFixed(2)}×`;
const FR = N.agreement.frontier;
const TOP3 = N.agreement.top3;
const TOP3_SHARED = sameSet(TOP3.ours, TOP3.eci) && sameSet(TOP3.ours, TOP3.aa);
const TOP3_COMMON = TOP3.ours.filter(m => TOP3.eci.includes(m) && TOP3.aa.includes(m));
const rest = (xs: string[]) => andList(xs.filter(m => !TOP3_COMMON.includes(m)));
const FRACTION = ['', '', 'half', 'third', 'quarter', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'];
const LEAD_GAP_POINTS = 15 * Math.abs(N.agreement.top3_theta[TOP3.ours[0]] - (N.agreement.top3_theta[TOP3.eci[0]] ?? N.agreement.top3_theta[TOP3.ours[0]]));

const FAQ: { id: string; q: string; a: React.ReactNode }[] = [
  {
    id: 'rank-agreement',
    q: 'What does “rank agreement” mean?',
    a: <>Take every pair of models and check whether two rankings put them in the same order. Rank (pairwise) agreement is the share of pairs where they do. 50% is chance, 100% is an identical ranking; it equals (Kendall&nbsp;τ&nbsp;+&nbsp;1)/2. We report it because it answers the question most readers care about: “is model A ahead of model B?”</>,
  },
  {
    id: 'tokens',
    q: 'How do you count tokens?',
    a: <>From the evaluation logs: the evaluated model’s input, output and reasoning tokens on every item it ran. The side of a run we did not do (the full suite for a core run, the core subset for a full run) is estimated per benchmark from the measured tokens per item and marked with a tilde (~). Costs are these token counts at list prices. Tokens used by judge models, which grade some benchmarks, are not counted on either side: judges are typically much cheaper models, so counting their tokens would understate the saving in dollars.</>,
  },
  {
    id: 'intervals',
    q: 'How are the 90% intervals computed?',
    a: <>With a parametric bootstrap (200 draws): the model’s ability is redrawn given its answers, with the calibration held fixed. The intervals therefore reflect which items a model answered, but not the error of the calibration or of the IRT model itself. For models held out of calibration, these raw intervals contained the full-run score {pct(IDX.capability.coverage_90)} of the time for Capability, but only {pct(IDX.reliability.coverage_90)} for Reliability and {pct(IDX.safety.coverage_90)} for Safety (section 3). We therefore widen the published Reliability intervals by {scale(SCALE.reliability)} and the Safety intervals by {scale(SCALE.safety)}, so that at least 90% of those models fall inside; Capability is left as is. The factors are fitted on the same held-out models, so expect real coverage to be slightly lower.</>,
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
  const method = METHOD;
  return (
    <article className="frame pt-22 pb-12">
      <div className="cols-8">
      <header>
        <Eyebrow>How it works</Eyebrow>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">How ComplAI CORE works</h1>
        <p className="mt-3 font-serif text-xl text-ink/80">Regulation-grounded LLM evaluation, predicted from a 2,000-item core.</p>
        <div className={`mt-8 ${P}`}>
          <p>
            In 2024 we published COMPL-AI<Cite ids={['ref-complai']} />, the first technical interpretation of the EU AI Act for large language models: it translated the Act’s broad principles into measurable requirements and a benchmark suite to test them. ComplAI CORE is the follow-up to this work. It keeps the same regulatory roots and runs {N.panel.benchmarks} benchmarks ({int(N.panel.items)} items) through one open harness, but any new model only has to answer a {int(N.panel.core_items)}-item core subset. From those answers (and our prior collection of full runs), we can then predict how it would do on the full suite, at a fraction of the cost.
          </p>
          <p>Next, we explain why we decided to build ComplAI CORE, how the benchmark reduction works, and how well it holds up.</p>
        </div>
      </header>

      <Figure
        id="figure-method"
        n={1}
        wide
        title="How CORE works: calibrate once on full runs, then score each new model from the core alone"
        caption={<>Schematic; the small grids and bars are illustrative, not data. Top: the calibration panel’s full runs are used once to fit item parameters and choose the core. Bottom: a new model only runs the core; its full-suite performance and index scores are predicted from those answers and the frozen parameters. Details in section 2.</>}
      >
        <MethodDiagram panel={N.panel} />
      </Figure>

      <div className="mt-14 space-y-14">
        <Section id="why" n={1} title="Why ComplAI CORE">
          <div className={P}>
            <p>As we see it, evaluating language models well is getting harder for four reasons:</p>
            <ul className="space-y-3 pl-5 list-disc marker:text-brand">
              <li>
                <strong>Cost and scale.</strong> New models arrive every week, and running a full suite on each is expensive: all {int(N.panel.items)} items on Claude Opus 5.5 cost about {usd(N.cost.opus.full)} at list prices, the core subset {usd(N.cost.opus.core)}.
              </li>
              <li>
                <strong>Reproducibility.</strong> Leaderboards often mix vendor-reported numbers from different prompts, settings and harnesses. Every CORE model runs through the same open, Inspect-based<Cite ids={['ref-inspect']} /> harness, with a public core list and calibration (<Faq id="reproduce">FAQ</Faq>).
              </li>
              <li>
                <strong>Benchmark weighting.</strong> A plain average counts a saturated or noisy benchmark as much as an informative one (resulting in frequent and somewhat arbitrary reweightings of leaderboards). Using item response theory (IRT), CORE weights each benchmark by how well it separates models.
              </li>
              <li>
                <strong>Regulatory grounding.</strong> The EU AI Act requires providers of general-purpose models with systemic risk to “perform model evaluation in accordance with standardised protocols and tools reflecting the state of the art” (Art.&nbsp;55(1)(a))<Cite ids={['ref-aiact']} />, and the Code of Practice<Cite ids={['ref-cop']} /> provides some details of what that involves. Most leaderboards connect to neither (section 6).
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
              <strong>Pool and panel.</strong> The suite has {N.panel.benchmarks} benchmarks with {int(N.panel.items)} items in total. We ran them in full on a panel of {N.panel.models} models, {N.panel.open_models} open-weight and {N.panel.closed_models} closed ({andList(N.panel.closed_names)}), resulting in about {(N.panel.responses / 1e6).toFixed(2)} million scored responses. We note that coverage is (partly) incomplete, i.e., not every model ran every benchmark; the closed models in particular ran only some benchmarks in full. The number of panel models per benchmark ranges from {N.panel.models_per_benchmark.min} ({task(N.panel.models_per_benchmark.min_task)}) to all {N.panel.models_per_benchmark.max}, with a median of {N.panel.models_per_benchmark.median}.
            </p>
            <p>
              <strong>Calibration.</strong> For each benchmark we fit a two-parameter logistic (2PL) IRT model<Cite ids={['ref-irt']} />, giving the probability that a model answers an item correctly from three numbers:
            </p>
            <div className="my-5 flex items-center justify-center gap-x-2 font-serif text-lg text-ink">
              <span><i>P</i>(correct)&nbsp;=&nbsp;<i>σ</i>(<i>α</i>&thinsp;(<i>θ</i>&nbsp;−&nbsp;<i>β</i>))</span>
            </div>
            <dl className="grid grid-cols-[2rem_1fr] gap-x-3 gap-y-1.5 pl-1">
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">θ</dt>
              <dd><strong>Model ability.</strong> Higher θ, more items answered correctly.</dd>
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">β</dt>
              <dd><strong>Item difficulty.</strong> Where on the ability scale an item becomes more likely right than wrong.</dd>
              <dt className="font-serif text-lg font-extrabold leading-snug text-brand-strong">α</dt>
              <dd><strong>Item discrimination.</strong> How sharply the item separates models just below that point from those just above.</dd>
            </dl>
            <p>
              <strong>The core subset.</strong> Our goal is to predict how a model would do on all {int(N.panel.items)} items while running only a small, fixed core, to cut cost and evaluation time. We do <em>not</em> simply average the model’s score on the core. Instead, we use its core answers to estimate its ability θ on each benchmark, and from θ we predict its result on every item of the full suite. The core therefore needs items where models differ most, spread fairly across the three indices. Each benchmark <i>b</i> gets a target <i>t</i><sub><i>b</i></sub>&nbsp;∝&nbsp;<i>σ</i><sub><i>b</i></sub><sup>2/3</sup>, where <i>σ</i><sub><i>b</i></sub> is how much models differ on it (with a minimum per benchmark). To keep the allocation balanced across benchmarks and indices, a joint allocator then hands out the 2,000 items one at a time, each to the benchmark maximising <span className="whitespace-nowrap"><i>t</i><sub><i>b</i></sub>&nbsp;/&nbsp;((<i>n</i><sub><i>b</i></sub>+1)(<i>n</i><sub><i>I</i></sub>+1))</span>, where <i>n</i><sub><i>b</i></sub> and <i>n</i><sub><i>I</i></sub> count the items already given to the benchmark and its index. Items within a benchmark are drawn by seeded random sampling, and the set is fixed, so every model answers the same items.
            </p>
            <p>
              <strong>Scoring a new model.</strong> A model answers the core items, and we estimate its ability on each benchmark. Its predicted benchmark score combines its real answers on the core items with the predicted probability of a correct answer on every other item: <span className="whitespace-nowrap">score&nbsp;=&nbsp;(Σ<sub>core</sub>&nbsp;real answers&nbsp;+&nbsp;Σ<sub>rest</sub>&nbsp;<i>P</i>(correct))&nbsp;/&nbsp;<i>N</i></span>, where <i>N</i> is the number of items in the benchmark.
            </p>
            <p>
              <strong>Indices.</strong> The benchmark scores are combined into three indices (Capability, Reliability and Safety) with a second IRT fit, in which each benchmark is one observation weighted by its fitted discrimination. A benchmark on which all models score alike has low discrimination and barely moves the result, while one that spreads models apart carries more weight. This naturally puts the emphasis on high-signal benchmarks and keeps our rankings in line with external leaderboards, without any manual weighting of benchmarks. Following Epoch AI’s ECI, the benchmark discriminations within an index get a log-normal prior (standard deviation 0.5 on the log scale) around their common average, so no single benchmark’s weight runs away or collapses to zero. Each index is published as <strong>100&nbsp;+&nbsp;15θ</strong> with a 90% interval.
            </p>
            <p id="labeling" className="scroll-mt-24">
              <strong>Evaluation areas.</strong> An index contains whole benchmarks, but evaluation areas are assigned per item. We labeled every item in the {N.panel.benchmarks} benchmarks with one of the 19 evaluation areas of our taxonomy, using Gemini 3.1 Pro, and Claude for some safety datasets. The labeler could only choose labels from the taxonomy, and gave a short rationale and a confidence level for each item. Both the model labels and the human reviews are <a href={`${REPO_URL}/tree/main/tools/label/labels`} className="font-semibold text-link hover:underline">published in the repository</a>. Evaluation areas do not enter the index scores. They report results by area (section 7) and link them to regulation (section 6).
            </p>
            <p>
              <strong>Smaller benchmarks and predictions.</strong> Compact benchmarks such as the Harbor Index<Cite ids={['ref-harbor']} /> curate a small set of hard, audited tasks and treat that set as the benchmark: the score is the score on those tasks. CORE works differently. Its 2,000 items are not picked for difficulty or audited one by one but represent a statistically allocated sample whose job is to predict a model’s result on the existing {int(N.panel.items)}-item suite. That makes every CORE score an estimate of full-suite performance, which we can check against real full runs (section 3).
            </p>
          </div>
        </Section>

        <Section id="accuracy" n={3} title="CORE accuracy">
          <div className={P}>
            <p>
              To test the accuracy of our aggregated IRT fits, we ran a leave-one-model-out experiment over all {N.lomo.models} models with full runs: remove one model, refit everything on the other {N.lomo.models - 1}, predict the removed model from its core answers only, and compare with its real full-suite result.
            </p>
            <p>
              As CORE publishes no single overall score, we first check something simple: a model’s accuracy across all the items it ran, predicted from its core answers versus actually achieved. Over the {N.lomo.models} held-out models, the mean absolute error (MAE) is <strong>{pct(N.lomo.overall_mae, 1)}</strong> ({pct(N.lomo.overall_mae_in_sample, 1)} when the model is part of calibration), and {pct(Math.min(IDX.capability.masked_mae, IDX.reliability.masked_mae, IDX.safety.masked_mae), 1)}–{pct(Math.max(IDX.capability.masked_mae, IDX.reliability.masked_mae, IDX.safety.masked_mae), 1)} within each index. In other words, from 2,000 items we predict a model’s result on all {int(N.panel.items)} to within about one to two percentage points. Single benchmarks are harder: per model and benchmark the MAE is {pct(N.lomo.task_mae_mean, 1)} (median {pct(N.lomo.task_mae_median, 1)}), highest on {task(N.lomo.worst_task)} at {pct(N.lomo.worst_task_mae, 1)}.
            </p>
            <p>
              On the published index scale, the MAE is {IDX.capability.score_mae.toFixed(1)} points for Capability, {IDX.reliability.score_mae.toFixed(1)} for Reliability and {IDX.safety.score_mae.toFixed(1)} for Safety, and no prediction is off by more than {Math.max(IDX.capability.score_max_ae, IDX.reliability.score_max_ae, IDX.safety.score_max_ae).toFixed(1)} points (Figure&nbsp;2). Rankings survive almost intact: predicted and full-run scores order <strong>{pct(IDX.capability.pairwise, 1)}</strong>, <strong>{pct(IDX.reliability.pairwise, 1)}</strong> and <strong>{pct(IDX.safety.pairwise, 1)}</strong> of model pairs the same way.
            </p>
            <p>
              We also checked how often the bootstrap 90% intervals contain the full-run score: {pct(IDX.capability.coverage_90)} for Capability, close to the intended 90%, but only {pct(IDX.reliability.coverage_90)} for Reliability and {pct(IDX.safety.coverage_90)} for Safety, because they leave out the uncertainty of the calibration and of the IRT model itself (<Faq id="intervals">FAQ</Faq>). We therefore publish Reliability intervals {scale(SCALE.reliability)} and Safety intervals {scale(SCALE.safety)} as wide, which brings their coverage in this test to {pct(covered('reliability'))} and {pct(covered('safety'))} (Figure&nbsp;2). Only {N.panel.closed_models} closed models, with partial full runs, are part of this test.
            </p>
          </div>
          <Figure
            n={2}
            wide
            title="Predicted from the core subset vs the full-suite score, for models held out of calibration"
            caption={<>Leave-one-model-out over {N.lomo.models} models with full-suite runs ({N.panel.open_models} open-weight, {N.panel.closed_models} closed with partial full runs). Capability n&nbsp;=&nbsp;{IDX.capability.n}, Reliability n&nbsp;=&nbsp;{IDX.reliability.n}, Safety n&nbsp;=&nbsp;{IDX.safety.n}; a model missing from an index ran too few of its benchmarks for an index score. Axes are on the published index scale. Bars are the published 90% intervals (Reliability widened {scale(SCALE.reliability)}, Safety {scale(SCALE.safety)}); hollow points fall outside theirs. Dashed line: perfect prediction. Each fold rescales θ slightly (by at most {N.lomo.max_drift_theta.toFixed(2)}θ), which is why we lead with accuracy errors above.</>}
          >
            <LomoFigure rows={LOMO} />
          </Figure>
          <p className={P}>
            <strong>In short:</strong> for a model like those in the panel, a core run lands on average within {IDX.capability.score_mae.toFixed(1)} points of its full-suite score for Capability, {IDX.reliability.score_mae.toFixed(1)} for Reliability and {IDX.safety.score_mae.toFixed(1)} for Safety, and rarely changes its order against another model. The weak spot is single benchmarks.
          </p>
        </Section>

        <Section id="agreement" n={4} title="Agreement with other leaderboards">
          <div className={P}>
            <p>
              Our Capability index orders models the same way as Epoch’s Capabilities Index (ECI)<Cite ids={['ref-eci']} /> for <strong>{pct(N.agreement.eci.pra.est)}</strong> of model pairs [95% CI {ci(N.agreement.eci.pra)}] across the {N.agreement.eci.n} models both cover, and the gaps between models match too (Pearson {N.agreement.eci.pearson.est.toFixed(2)} [{N.agreement.eci.pearson.lo.toFixed(2)}–{N.agreement.eci.pearson.hi.toFixed(2)}]). Against the Artificial Analysis Intelligence Index v4<Cite ids={['ref-aa']} /> it is {pct(N.agreement.aa.pra.est)} [{ci(N.agreement.aa.pra)}] over {N.agreement.aa.n} models. Those two leaderboards agree with each other on {pct(N.agreement.ceiling.pra.est)} [{ci(N.agreement.ceiling.pra)}] of the {N.agreement.ceiling.n} models both cover (<Faq id="rank-agreement">FAQ</Faq>). In other words, we rank models about as consistently with ECI as the two leaderboards rank them with each other.
            </p>
            <p>
              Disagreements are mostly where ECI itself is unsure: we agree on {pct(N.agreement.eci_pairs.separated_agree)} of the pairs its error bars separate and {pct(N.agreement.eci_pairs.overlap_agree)} of those where they overlap.{' '}
              {TOP3_SHARED ? (
                <>All three rankings have the same top three: {andList(TOP3.ours)}. {TOP3.ours[0] === TOP3.eci[0] ? <>All put {TOP3.ours[0]} first.</> : <>We put {TOP3.ours[0]} first by {LEAD_GAP_POINTS.toFixed(2)} points, well inside both intervals, so it is effectively a tie; both leaderboards put {TOP3.eci[0]} first.</>}</>
              ) : TOP3_COMMON.length >= 2 ? (
                <>The same models lead everywhere: {andList(TOP3_COMMON)} are in all three top threes, and only the remaining place differs ({rest(TOP3.ours)} for us, {rest(TOP3.eci)} for ECI, {rest(TOP3.aa)} for Artificial Analysis).</>
              ) : (
                <>The top three differ: ours is {andList(TOP3.ours)}; ECI’s {andList(TOP3.eci)}; Artificial Analysis’ {andList(TOP3.aa)}.</>
              )}
            </p>
            <p>
              <strong>Frontier check.</strong> Close models are harder to order, so we repeat the comparison on the {FR.k} strongest models we share with ECI. We agree on {pct(FR.pra.est)} [{ci(FR.pra)}] of pairs, and on {pct(FR.separated_agree)} of the pairs ECI itself can separate: when ECI is sure which model is stronger, so are we. The gaps carry over too (Pearson {FR.pearson.est.toFixed(2)} [{FR.pearson.lo.toFixed(2)}–{FR.pearson.hi.toFixed(2)}]), so we keep not just the order but the distances. Only {FR.k - N.agreement.eci_top15_missing} of ECI’s top {FR.k} are in our set so far.
            </p>
            <p>
              We note that this is agreement and not accuracy. The rankings measure overlapping but different things: AA includes agentic tasks, ECI aggregates many published benchmarks, and our Capability index uses 8 benchmarks on a core subset.
            </p>
          </div>
          <Figure
            n={3}
            wide
            title="ComplAI CORE Capability vs Epoch Capabilities Index"
            caption={<>{N.agreement.eci.n} models covered by both; bars are our 90% intervals and ECI’s 90% intervals. ECI data pulled {method._meta.eci_pulled}. Confidence intervals in the text come from a 2,000-draw bootstrap over models.</>}
          >
            <EciFigure rows={method.eci} frontier={FR.k} cited={[...TOP3.ours, ...TOP3.eci]} />
          </Figure>
        </Section>

        <Section id="cost" n={5} title="Costs">
          <div className={P}>
            <p>
              At list prices, a core run costs a median of <strong>{times(N.cost.openrouter.median)} less</strong> than a full run (about {pct(1 - 1 / N.cost.openrouter.median)} less) across {N.cost.openrouter.n} models priced on OpenRouter, ranging from {times(N.cost.openrouter.min)} to {times(N.cost.openrouter.max)}. For the {N.cost.direct.n} closed models at their providers’ own prices, the median is <strong>{times(N.cost.direct.median)}</strong> ({N.cost.direct.min.toFixed(1)}–{times(N.cost.direct.max)}). In tokens, a typical core run uses {pct(N.cost.token_reduction)} fewer. In short, evaluating a new model costs about a {FRACTION[Math.round(N.cost.openrouter.median)] ?? `1/${Math.round(N.cost.openrouter.median)}`} of a full run.
            </p>
            <p>
              These figures count the evaluated model’s input, output and reasoning tokens, while judge-model calls are excluded on both sides (<Faq id="tokens">FAQ</Faq>). We note that these are list prices and can differ in practice (e.g., our open-weight runs were on our own servers), so OpenRouter prices stand in for what an average user would pay. Generally, batch APIs halve both sides and leave the ratio unchanged.
            </p>
          </div>
          <Figure
            n={4}
            title="One evaluation run in USD: core subset vs full suite"
            caption={<>Representative models. Each run’s cost is measured on the items it actually ran and extrapolated to the rest per benchmark from measured tokens per item: core-only closed models have an extrapolated full-suite cost, open-weight models with full runs an extrapolated core cost, and the Claude models with partial full runs a mix of both. Prices pulled {method._meta.prices_pulled}<Cite ids={['ref-prices']} />. Judge tokens excluded.</>}
          >
            <CostFigure rows={method.cost} items={N.panel.items} />
          </Figure>
        </Section>

        <Section id="regulation" n={6} title="Regulations">
          <div className={P}>
            <p>
              CORE inherits its structure from COMPL-AI<Cite ids={['ref-complai']} />. Its three indices and 19 evaluation areas each cite the EU AI Act articles<Cite ids={['ref-aiact']} /> and Code of Practice measures<Cite ids={['ref-cop']} /> they inform. Where the taxonomy covers a risk our current benchmarks don’t measure, we aim to be open about it and explicitly mark the gap rather than hide it.
            </p>
            <p>
              The two most-cited articles have different scopes. <strong>Article&nbsp;15</strong> sets accuracy, robustness, and cybersecurity requirements for <em>high-risk AI systems</em>. We use it as a technical reference for what to measure: factuality, consistency and perturbation robustness point to its accuracy and robustness requirements (Art.&nbsp;15(1)), and jailbreak and prompt-injection resistance to its requirement of resilience against attempts to exploit system vulnerabilities (Art.&nbsp;15(5)). <strong>Article&nbsp;55</strong> sets obligations for <em>providers of general-purpose AI models with systemic risk</em>, including model evaluation with adversarial testing (Art.&nbsp;55(1)(a)) and adequate cybersecurity protection (Art.&nbsp;55(1)(d)); the Code of Practice details how to meet them. Four Safety evaluation areas and perturbation robustness cite both, since the same measurement informs a system-level requirement and a model-level obligation.
            </p>
            <p>
              While the citations explain <em>why</em> we measure something, we note that they are not a compliance verdict (<Faq id="compliance">FAQ</Faq>). The full mapping, with the cited text, is on the <Link href="/mapping" className="font-semibold text-link hover:underline">AI Act mapping</Link>.
            </p>
          </div>
        </Section>

        <Section id="reading" n={7} title="Interpretation of scores">
          <div className={P}>
            <p>
              <strong>The scale.</strong> Each index is published as 100&nbsp;+&nbsp;15θ, where θ is standardized over the calibration panel. 100 is the panel average and 15 points is one standard deviation, so 115 means one standard deviation above average. Scores compare models with each other and should therefore not be read as percentages. Every score has a 90% interval. When two intervals overlap substantially, the data may not separate the two models. If too few of an index’s benchmarks were run, no score is shown.
            </p>
            <p>
              <strong>Run types.</strong> Every result is tagged with how it was produced. A <strong>Calibration</strong> run evaluated every item in the benchmarks it ran; these runs form the panel on which the item parameters are fitted, and most cover only some benchmarks. A <strong>Core</strong> run answered only the core items. A few runs are <strong>Mixed</strong>: some benchmarks in full, others on the core subset only. A benchmark whose core subset is close to its full size does not decide the tag. All runs are scored the same way, from their core answers, so every model is compared on equal terms.
            </p>
          </div>

          <div id="areas" className={`scroll-mt-24 ${P}`}>
            <h3 className="pt-2 text-lg font-bold text-ink">Evaluation-area scores</h3>
            <p>
              An index always contains whole benchmarks, so it can be fitted on benchmark scores. Evaluation areas (for example math or coding within Capability) are <a href="#labeling" className="font-semibold text-link hover:underline">labeled per item</a> and cut across benchmarks, so they are not put on the index scale. Instead, each shows the model’s <strong>predicted accuracy</strong> on all of its items: real answers on the CORE items plus predictions for the rest. Each benchmark’s accuracy on its items in the area is weighted the same way as in the index, by how well it separates models (its discrimination), times the share of its own items that fall in the area. A large benchmark therefore cannot drown out the others just by having more items: in Fairness and bias, BBQ has 58,492 items and FairLLM 200, but they weigh about the same. Predicted accuracy depends on how hard the items are, so they compare models within an evaluation area but not across areas. The model pages mark the median model for reference.
            </p>
            <p>
              Each evaluation-area score has a 90% interval, from a bootstrap that redraws the model’s ability on each benchmark within its uncertainty. If a model skipped some of an area’s benchmarks, those items are estimated from the rest of its index, and that extra error is not in the interval.
            </p>
          </div>

          <div id="efficiency" className={`scroll-mt-24 ${P}`}>
            <h3 className="pt-2 text-lg font-bold text-ink">Token estimates</h3>
            <p>
              {efficiency ? (
                <>
                  The core subset has {efficiency.subset_items.toLocaleString()} items, against {efficiency.population_items.toLocaleString()} in the full suite, about {Math.round(efficiency.population_items / efficiency.subset_items)}&times; fewer.
                  {efficiency.token_reduction !== null && (
                    <> Measured in tokens, the saving is smaller, about {formatPercent(efficiency.token_reduction)}: a typical evaluation uses roughly {formatTokens(efficiency.core_tokens_median)} tokens on the core subset against {formatTokens(efficiency.full_tokens_median)} on the full suite.</>
                  )}
                </>
              ) : (
                <>The core subset is a small fraction of the full suite, so a core run costs a fraction of running every item.</>
              )}
              {' '}The gap exists because the core subset keeps most items from the long, token-heavy benchmarks (multi-step math, jailbreak and rule-following dialogues), where each item says a lot about a model, and drops most items from short multiple-choice benchmarks.
            </p>
            <p>
              Token counts come from the evaluation logs and include the evaluated model’s input, output, and reasoning tokens, excluding judge calls. For each run, we measure the tokens of the run actually done and <strong>estimate</strong> the other figure by scaling each benchmark’s measured tokens per item to the other item count, and mark it with a tilde (~). Suite-wide figures use the typical tokens per item of each benchmark across all models (average input, median output). Runs whose logs have no token counts show a dash.
            </p>
          </div>
        </Section>

        <Section id="enables" n={8} title="What this enables">
          <div className={P}>
            <p>Returning to the four problems outlined at the top, we stand roughly here:</p>
            <ul className="space-y-3 pl-5 list-disc marker:text-brand">
              <li>
                <strong>Cost and scale.</strong> A core run costs a median of {times(N.cost.openrouter.median)} less than a full run, {times(N.cost.direct.median)} for closed models at their providers’ prices (section 5).
              </li>
              <li>
                <strong>Reproducibility.</strong> One open harness, a fixed public core list and public item parameters, so anyone can run CORE and get scores comparable to ours.
              </li>
              <li>
                <strong>Benchmark weighting.</strong> Benchmarks count by how well they separate models, and the result reproduces full-suite accuracy to {pct(N.lomo.overall_mae, 1)} MAE on held-out models while agreeing with ECI on {pct(N.agreement.eci.pra.est)} of model pairs. Capability, Safety and Reliability are reported separately, never folded into one number.
              </li>
              <li>
                <strong>Regulatory grounding.</strong> All 19 evaluation areas cite the EU AI Act articles and Code of Practice measures they inform, with gaps marked.
              </li>
            </ul>
            <p>
              In practice, this means more models can be evaluated more often: on every release rather than a few times a year, with Capability, Reliability, and Safety tracked over time on the same footing.
            </p>
            <p>
              <strong>Limitations.</strong> We note that CORE predicts the existing suite, but it does not validate it. If a benchmark or its scorer is flawed, CORE inherits the flaw. Benchmark saturation remains an open problem: discrimination weighting lowers the influence of benchmarks that no longer separate models, but it does not make them informative again. Held-out validation covers open-weight models and only {N.panel.closed_models} closed models, each with partial full runs. The 90% intervals reflect which questions were sampled from each benchmark; they do not include uncertainty from the choice of benchmarks in an index, which is largest for Safety.
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
