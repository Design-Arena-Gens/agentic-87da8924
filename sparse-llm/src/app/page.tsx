"use client";

import { useEffect, useMemo, useState } from "react";
import { SparseLLM, SparseRunResult } from "@/lib/sparseLLM";

const DEFAULT_PROMPT =
  "Design a sparse transformer that routes tokens to only the neurons they need.";

type ActivationHeatProps = {
  label: string;
  intensity: number;
  weight: number;
};

function ActivationHeat({ label, intensity, weight }: ActivationHeatProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2 backdrop-blur transition hover:bg-white/10">
      <div className="flex flex-col">
        <span className="text-sm font-medium text-white">{label}</span>
        <span className="text-xs text-white/70">
          routing prob: {(weight * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex h-2 w-24 items-center rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-emerald-400 transition-all"
          style={{ width: `${Math.min(100, intensity * 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function Home() {
  const model = useMemo(() => new SparseLLM(), []);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [result, setResult] = useState<SparseRunResult>(() => model.run(prompt));

  useEffect(() => {
    setResult(model.run(prompt));
  }, [model, prompt]);

  const description = model.describe();

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.2)_0,_rgba(15,23,42,0)_70%)]" />
      <main className="relative mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 pb-20 pt-16 sm:px-10 lg:px-16">
        <header className="flex flex-col gap-4">
          <span className="inline-flex w-fit items-center gap-2 rounded-full border border-white/10 bg-white/10 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-200">
            Sparse Mixture Routing
          </span>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
            Truly sparse LLM architecture that fires only the neurons required.
          </h1>
          <p className="max-w-3xl text-lg leading-relaxed text-white/70">
            This simulator demonstrates an adaptive mixture-of-experts transformer
            stack with deterministic routing. Each token dynamically activates a
            compact set of neurons, yielding efficient inference while preserving
            contextual capacity.
          </p>
        </header>

        <section className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-emerald-500/10 backdrop-blur lg:p-8">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-2xl font-semibold">Prompt</h2>
                <p className="text-sm text-white/60">
                  Modify the prompt to inspect routing patterns and neuron
                  utilization under different linguistic demands.
                </p>
              </div>
              <button
                type="button"
                className="rounded-full border border-emerald-400/40 bg-emerald-500/20 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-emerald-100 transition hover:bg-emerald-500/30"
                onClick={() => setPrompt(DEFAULT_PROMPT)}
              >
                Reset
              </button>
            </div>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              className="min-h-[140px] rounded-2xl border border-white/10 bg-black/30 px-4 py-3 font-mono text-sm text-emerald-100 shadow-inner focus:border-emerald-400 focus:outline-none focus:ring-0"
              spellCheck={false}
            />
            <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-emerald-200">
                Predicted continuation token
              </h3>
              <div className="mt-2 flex items-center gap-3">
                <code className="rounded-full bg-white/10 px-4 py-2 text-lg text-emerald-200">
                  {result.predictedToken}
                </code>
                <span className="text-xs text-white/50">
                  highest logit across {description.vocabulary} learned embeddings
                </span>
              </div>
            </div>
          </div>

          <aside className="flex flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-cyan-500/10 backdrop-blur lg:p-8">
            <h2 className="text-xl font-semibold text-cyan-200">
              Architecture snapshot
            </h2>
            <div className="grid grid-cols-2 gap-3 text-sm text-white/80">
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Layers
                </p>
                <p className="mt-1 text-2xl font-semibold">{description.layers}</p>
              </div>
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Neurons per layer
                </p>
                <p className="mt-1 text-2xl font-semibold">
                  {description.neuronsPerLayer}
                </p>
              </div>
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Active neurons
                </p>
                <p className="mt-1 text-2xl font-semibold">{description.topK}</p>
              </div>
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Hidden width
                </p>
                <p className="mt-1 text-2xl font-semibold">
                  {description.hiddenDim}
                </p>
              </div>
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Embedding dim
                </p>
                <p className="mt-1 text-2xl font-semibold">
                  {description.embeddingDim}
                </p>
              </div>
              <div className="rounded-2xl bg-black/30 px-4 py-3">
                <p className="text-xs uppercase tracking-wide text-white/50">
                  Vocabulary
                </p>
                <p className="mt-1 text-2xl font-semibold">
                  {description.vocabulary}
                </p>
              </div>
            </div>
            <p className="text-xs leading-5 text-white/60">
              Deterministic routing guarantees only {description.topK} experts
              fire per layer. Traces below expose the exact neuron subset per
              token, illustrating sparsity across the stack.
            </p>
          </aside>
        </section>

        <section className="rounded-3xl border border-white/10 bg-black/40 p-6 shadow-2xl shadow-purple-500/10 backdrop-blur md:p-8">
          <div className="flex flex-col gap-2 pb-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold">
                Sparse firing timeline
              </h2>
              <p className="text-sm text-white/60">
                Token-by-token neuron routing. Width of each highlight encodes
                expert contribution weight.
              </p>
            </div>
            <span className="text-xs uppercase tracking-[0.3em] text-purple-200">
              Layered expert selection
            </span>
          </div>
          <div className="flex flex-col gap-6">
            {result.tokenTraces.map((trace) => (
              <div
                key={`${trace.token}-${trace.tokenIndex}`}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 md:p-5"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className="rounded-full bg-purple-500/20 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-purple-100">
                      Token #{trace.tokenIndex}
                    </span>
                    <code className="rounded-full bg-white/10 px-3 py-1 text-sm text-purple-100">
                      {trace.token}
                    </code>
                  </div>
                  <span className="text-[10px] uppercase tracking-[0.25em] text-white/50">
                    {trace.layers.length} layers
                  </span>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                  {trace.layers.map((layer) => (
                    <div
                      key={`${trace.token}-${layer.layerIndex}`}
                      className="flex flex-col gap-3 rounded-xl border border-white/10 bg-black/30 p-3"
                    >
                      <div className="flex items-center justify-between text-xs text-white/50">
                        <span>Layer {layer.layerIndex + 1}</span>
                        <span>
                          {layer.selected.length} / {description.neuronsPerLayer} active
                        </span>
                      </div>
                      <div className="flex flex-col gap-2">
                        {layer.selected.map((item) => (
                          <ActivationHeat
                            key={`${layer.layerIndex}-${item.id}`}
                            label={`Neuron ${item.id}`}
                            intensity={Math.min(1, Math.abs(item.rawScore) / 6)}
                            weight={item.weight}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-blue-500/10 backdrop-blur md:p-8">
          <h2 className="text-2xl font-semibold text-blue-100">
            Design principles
          </h2>
          <ul className="mt-4 grid gap-4 md:grid-cols-2">
            <li className="rounded-2xl bg-black/30 p-4">
              <h3 className="text-lg font-semibold text-blue-100">
                Deterministic router
              </h3>
              <p className="mt-2 text-sm text-white/70">
                Per-token gating network computes scores over {description.neuronsPerLayer} experts.
                Only the top-{description.topK} neurons fire, with softmaxed routing coefficients to
                ensure normalized contribution.
              </p>
            </li>
            <li className="rounded-2xl bg-black/30 p-4">
              <h3 className="text-lg font-semibold text-blue-100">
                Layer-wise residual memory
              </h3>
              <p className="mt-2 text-sm text-white/70">
                Layer outputs are normalized and fused into the persistent hidden
                state, allowing sparse experts to accumulate context without dense
                activation overhead.
              </p>
            </li>
            <li className="rounded-2xl bg-black/30 p-4">
              <h3 className="text-lg font-semibold text-blue-100">
                Embedding-aligned logits
              </h3>
              <p className="mt-2 text-sm text-white/70">
                Final prediction uses an embedding-similarity projection so token
                vectors double as classifier weights, preserving efficiency when
                the vocabulary grows.
              </p>
            </li>
            <li className="rounded-2xl bg-black/30 p-4">
              <h3 className="text-lg font-semibold text-blue-100">
                Inspectable sparsity
              </h3>
              <p className="mt-2 text-sm text-white/70">
                The routing trace exposes which experts fire per layer, enabling
                interpretability tooling and pruning strategies tuned to observed
                utilization.
              </p>
            </li>
          </ul>
        </section>
      </main>
    </div>
  );
}
