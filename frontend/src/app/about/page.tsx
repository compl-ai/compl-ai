export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 max-w-4xl py-12 space-y-12">
      <div>
        <h1 className="text-4xl font-bold tracking-tight mb-4">About COMPL-AI</h1>
        <p className="text-xl text-muted-foreground">
          Project context, motivation, and grounding.
        </p>
      </div>

      <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
        <section>
          <h2>The COMPL-AI Framework</h2>
          <p>
            COMPL-AI is designed to provide high-signal, rigorous evaluation of AI models and agents. Our goal is to move beyond superficial benchmark reporting and provide deep insights into a system's true capabilities and risks.
          </p>
        </section>

        <section>
          <h2>EU AI Act Motivation</h2>
          <p>
            The architecture of COMPL-AI is heavily motivated by the European Union's Artificial Intelligence Act. As AI systems become more agentic and deeply integrated into critical infrastructure, there is an urgent need for evaluation frameworks that map technical capabilities directly to regulatory principles like Safety, Security, Reliability, and Fairness.
          </p>
        </section>

        <section>
          <h2>Contributors & Acknowledgements</h2>
          <p>
            The COMPL-AI framework is developed by an open coalition of researchers and engineers dedicated to robust AI evaluation.
          </p>
        </section>

        <section>
          <h2>Resources</h2>
          <ul>
            <li>
              <a href="https://github.com/compl-ai">GitHub Repository</a>
            </li>
            <li>
              <a href="/data/predictions.core.v1.json">core.v1 Dataset</a>
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
