import { ArrowRight } from "lucide-react";

export default function MethodologyPage() {
  return (
    <div className="container mx-auto px-4 max-w-4xl py-12 space-y-16">
      <div className="space-y-4 max-w-2xl">
        <h1 className="text-3xl font-medium tracking-tight text-gray-900">Methodology</h1>
        <p className="text-gray-500 text-sm">
          How COMPL-AI translates a massive evaluation pool into highly-accurate predicted performance via Item Response Theory (IRT).
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-gray-50 rounded-sm border border-gray-100">
          <div className="text-xs font-bold text-gray-400 mb-1">STEP 1</div>
          <h3 className="font-medium text-sm text-gray-900">Evaluation Pool</h3>
          <p className="text-xs text-gray-500 mt-1">Tens of thousands of raw test items.</p>
        </div>
        <div className="p-4 bg-gray-50 rounded-sm border border-gray-100">
          <div className="text-xs font-bold text-gray-400 mb-1">STEP 2</div>
          <h3 className="font-medium text-sm text-gray-900">IRT Calibration</h3>
          <p className="text-xs text-gray-500 mt-1">Fitting 2PL parameters for every item.</p>
        </div>
        <div className="p-4 bg-primary/10 rounded-sm border border-primary/20">
          <div className="text-xs font-bold text-primary/70 mb-1">STEP 3</div>
          <h3 className="font-medium text-sm text-primary">Core Subset</h3>
          <p className="text-xs text-primary/80 mt-1">Selecting the most informative items.</p>
        </div>
        <div className="p-4 bg-gray-900 rounded-sm border border-gray-800">
          <div className="text-xs font-bold text-gray-400 mb-1">STEP 4</div>
          <h3 className="font-medium text-sm text-white">Prediction</h3>
          <p className="text-xs text-gray-400 mt-1">Recovering full-dataset performance.</p>
        </div>
      </div>

      <div className="space-y-12">
        <section className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">1. Sample-level Coverage</h2>
          <div className="text-sm text-gray-600 space-y-4 leading-relaxed">
            <p>
              Each source benchmark is assigned as a whole to one of three indices: <strong>Capability, Reliability, and Safety</strong>. A benchmark only contributes to the index it is assigned to.
            </p>
            <p>
              Within an index, individual items are labelled with sub-categories (for example, math or coding within Capability). Sub-categories are used for reporting coverage only; a sub-category with too few items is marked as a gap and not scored.
            </p>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">2. IRT Model & Calibration</h2>
          <div className="text-sm text-gray-600 space-y-4 leading-relaxed">
            <p>
              We use a <strong>2-Parameter Logistic (2PL)</strong> Item Response Theory model. Unlike standard accuracy scores which weight all questions equally, IRT understands that not all questions are equally useful.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            <div className="space-y-1">
              <div className="text-xl font-serif text-primary">&theta; (Theta)</div>
              <h4 className="font-medium text-sm text-gray-900">Model Capability</h4>
              <p className="text-xs text-gray-500">The latent ability of the AI model being evaluated. Higher &theta; means the model is generally stronger.</p>
            </div>
            <div className="space-y-1">
              <div className="text-xl font-serif text-primary">&beta; (Beta)</div>
              <h4 className="font-medium text-sm text-gray-900">Item Difficulty</h4>
              <p className="text-xs text-gray-500">Where the question sits on the ability scale. Hard questions require high &theta; to answer correctly.</p>
            </div>
            <div className="space-y-1">
              <div className="text-xl font-serif text-primary">&alpha; (Alpha)</div>
              <h4 className="font-medium text-sm text-gray-900">Item Discrimination</h4>
              <p className="text-xs text-gray-500">How sharply the question distinguishes between models slightly above vs slightly below the difficulty threshold.</p>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900 border-b pb-2">3. Predicted Performance</h2>
          <div className="text-sm text-gray-600 space-y-4 leading-relaxed">
            <p>
              By testing a new model against our compact core subset, we can accurately estimate its &theta;. Using this &theta; and the known parameters of the tens of thousands of items in the total evaluation pool, we can calculate the expected probability of the model getting every single question right.
            </p>
            <p>
              The <strong>Predicted Score</strong> you see on the leaderboard is the population-item-weighted average of these probabilities. This allows us to predict a model's true capability without running it against the massive, expensive full dataset.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
