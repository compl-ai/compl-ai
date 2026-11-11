import type { Metadata } from 'next';
import Link from 'next/link';
import { Eyebrow } from '@/components/layout/Eyebrow';
import { README_CORE_URL, REPO_URL, pageMetadata } from '@/lib/site';
import { withBasePath } from '@/lib/utils';

export const metadata: Metadata = pageMetadata('About', 'Who builds ComplAI, how it started, and how to cite and contribute.');

const P = 'space-y-4 text-[15px] leading-relaxed text-ink/85';
const A = 'font-semibold text-link hover:underline';
const PAPER_URL = 'https://arxiv.org/abs/2410.07959';

const BIBTEX = `@article{complai24,
  title={COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act},
  author={Philipp Guldimann and Alexander Spiridonov and Robin Staab and Nikola Jovanovi\\'{c} and Mark Vero and Velko Vechev and Anna Gueorguieva and Mislav Balunovi\\'{c} and Nikola Konstantinov and Pavol Bielik and Petar Tsankov and Martin Vechev},
  year={2024},
  eprint={2410.07959},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.07959},
}`;

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24 space-y-4">
      <h2 className="text-2xl font-extrabold tracking-tight text-ink">{title}</h2>
      {children}
    </section>
  );
}

export default function AboutPage() {
  return (
    <article className="frame pt-22 pb-12">
      <div className="cols-8">
        <header>
          <Eyebrow>About</Eyebrow>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-ink">About ComplAI</h1>
        </header>

        <div className="mt-14 space-y-14">
          <Section id="history" title="History">
            <div className={P}>
              <p>
                In 2024, ETH Zurich (SRI Lab), INSAIT and LatticeFlow AI published{' '}
                <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className={A}>COMPL-AI</a>, the first technical
                interpretation of the EU AI Act for large language models. It translated the Act’s principles into measurable
                technical requirements and an open-source benchmark suite to test them.
              </p>
              <p>
                ComplAI CORE is the follow-up. It keeps the same regulatory roots, moves the suite to the open{' '}
                <a href="https://github.com/UKGovernmentBEIS/inspect_ai" target="_blank" rel="noopener noreferrer" className={A}>Inspect</a>{' '}
                harness, and scores models on three indices: Capability, Reliability and Safety. A new model only answers a
                2,000-item core subset, and its full-suite scores are predicted from those answers. See{' '}
                <Link href="/methodology" className={A}>How it works</Link> for the method and its validation, and the{' '}
                <Link href="/mapping" className={A}>AI Act mapping</Link> for how the evaluation areas link to the regulation.
              </p>
            </div>
          </Section>

          {/* TODO: team, SwissAI support and other acknowledgements, once confirmed. */}
          <Section id="team" title="Who we are">
            <div className={P}>
              <p>ComplAI is created and maintained by ETH Zurich (SRI Lab), INSAIT and LatticeFlow AI.</p>
            </div>
          </Section>

          <Section id="contribute" title="Contribute">
            <div className={P}>
              <p>
                The harness, the core item list and the calibrated item parameters are open source under the Apache 2.0
                license in the <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className={A}>GitHub repository</a>.
                The <a href={README_CORE_URL} target="_blank" rel="noopener noreferrer" className={A}>README</a> shows how to run
                the core subset on your own model and predict its scores. We welcome new benchmarks, fixes and extensions of
                the regulatory mapping; please open an issue or a pull request.
              </p>
            </div>
          </Section>

          <Section id="cite" title="Cite">
            <div className={P}>
              <p>
                If you use ComplAI, please cite the{' '}
                <a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className={A}>original paper</a>:
              </p>
            </div>
            <pre className="overflow-x-auto rounded-md border border-border bg-muted/50 p-4 text-xs leading-relaxed text-ink/85">
              <code>{BIBTEX}</code>
            </pre>
          </Section>

          <Section id="resources" title="Resources">
            <ul className={`${P} list-disc pl-5 marker:text-brand`}>
              <li><a href={REPO_URL} target="_blank" rel="noopener noreferrer" className={A}>GitHub repository</a></li>
              <li><a href={PAPER_URL} target="_blank" rel="noopener noreferrer" className={A}>COMPL-AI paper (arXiv:2410.07959)</a></li>
              <li><a href={withBasePath('/data/predictions.json')} className={A}>Raw predictions (JSON)</a></li>
            </ul>
          </Section>
        </div>
      </div>
    </article>
  );
}
