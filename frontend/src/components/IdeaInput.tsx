import { useState, useCallback, useRef } from "react";
import { motion } from "motion/react";
import SourceSelection from "./SourceSelection";
import type { EvidenceSelection } from "../types/api";

interface Props {
  onSubmit: (idea: string, selection: EvidenceSelection) => void;
  disabled: boolean;
}

type Specimen = {
  numeral: string;
  title: string;
  domain: string;
  provenance: string;
  body: string;
};

const SPECIMENS: Specimen[] = [
  {
    numeral: "i",
    title: "Attention Is All You Need",
    domain: "NLP · seq-to-seq",
    provenance: "Vaswani et al., 2017 — exists",
    body:
      "I propose a sequence transduction architecture that eschews recurrence and convolutions entirely, relying instead on stacked self-attention and pointwise feed-forward layers. The encoder maps an input sequence into a continuous representation by computing scaled dot-product attention over all token pairs in parallel, while the decoder generates output tokens autoregressively through masked self-attention and encoder-decoder attention. Multi-head attention projects queries, keys, and values into several subspaces so the model can jointly attend to information from different representation subspaces at different positions. Sinusoidal positional encodings are added to input embeddings to inject sequence order. The architecture is trained on WMT 2014 English-German and English-French translation using a warm-up learning rate schedule, dropout, and label smoothing, and is expected to achieve state-of-the-art BLEU scores while training significantly faster than recurrent or convolutional sequence models.",
  },
  {
    numeral: "ii",
    title: "Deep Residual Learning",
    domain: "Vision · ImageNet",
    provenance: "He et al., 2015 — exists",
    body:
      "I propose a residual learning framework for training extremely deep convolutional neural networks. The central idea is to reformulate each layer as learning a residual function with reference to the layer input, rather than learning unreferenced functions, by introducing identity shortcut connections that skip one or more layers and add the input directly to the output. These shortcuts add neither parameters nor computational complexity, yet they allow gradients to flow more directly during backpropagation, mitigating the degradation problem observed when naively increasing network depth. The architecture is composed of stacked residual blocks of convolution-batch normalization-ReLU triplets and is evaluated on ImageNet classification at depths up to 152 layers, COCO detection, and Pascal VOC segmentation. Empirical results should demonstrate that residual networks are easier to optimize and gain accuracy from substantially increased depth, resolving the accuracy saturation observed in plain deep networks.",
  },
  {
    numeral: "iii",
    title: "Bidirectional Transformer Pre-training",
    domain: "NLP · transfer learning",
    provenance: "Devlin et al., 2018 — exists",
    body:
      "I propose a bidirectional transformer encoder pre-trained on large unlabeled text corpora using two unsupervised objectives. The first is masked language modeling, where a random subset of input tokens is replaced with a [MASK] symbol and the model is asked to predict the original tokens from bidirectional context. The second is next sentence prediction, where the model determines whether a candidate second sentence actually follows the first in the source corpus. After pre-training on Wikipedia and BookCorpus with WordPiece tokenization, the resulting representations are fine-tuned with a single additional output layer for a wide range of downstream natural language understanding tasks, including question answering, named entity recognition, and natural language inference. Bidirectional conditioning, in contrast to left-to-right language models, is hypothesized to yield richer contextual representations and to set new state-of-the-art results on the GLUE benchmark, SQuAD, and SWAG without substantial task-specific architecture changes.",
  },
  {
    numeral: "iv",
    title: "Adversarial Generative Networks",
    domain: "Generative · likelihood-free",
    provenance: "Goodfellow et al., 2014 — exists",
    body:
      "I propose a generative modeling framework based on an adversarial process in which two neural networks are trained simultaneously: a generator network that maps samples from a simple prior noise distribution into the data manifold, and a discriminator network that learns to distinguish generated samples from real training examples. The two networks are trained in a minimax game where the generator seeks to maximize the probability that the discriminator misclassifies its outputs, while the discriminator seeks to minimize its classification error. At equilibrium the generator is expected to recover the underlying data distribution and the discriminator should output one half everywhere. Training proceeds by alternating gradient updates on each network using standard backpropagation, requiring neither Markov chains nor unrolled approximate inference. The framework is evaluated on MNIST, the Toronto Face Database, and CIFAR-10, and is expected to produce visually plausible samples, opening a new family of likelihood-free generative models.",
  },
  {
    numeral: "v",
    title: "Fractal-Attractor Continual Learning",
    domain: "AI/ML · continual",
    provenance: "novel — speculative",
    body:
      "I propose a continual learning framework in which catastrophic forgetting is reframed as a structured geometric organization problem rather than purely a regularization or replay problem. After each task is learned, the model deliberately reorganizes the previously acquired knowledge into a set of fractal attractors in weight space, leaving sparse low-curvature regions between attractors that remain available for subsequent learning. New incoming tasks are encouraged through curvature-aware optimization to settle into these unoccupied gaps rather than overwriting existing attractors, so that prior representations are preserved by virtue of where they sit in parameter space, not by replaying old data or penalizing weight movement. The fractal arrangement is hypothesized to provide an unbounded supply of self-similar empty pockets at multiple scales, enabling the system to keep absorbing new tasks without saturating capacity. The proposal differs from elastic weight consolidation, experience replay, and parameter isolation methods because forgetting is treated not as a failure to suppress but as a controlled compression that physically clears space for further learning.",
  },
  {
    numeral: "vi",
    title: "Stigmergic Cyanobacteria Albedo Rafts",
    domain: "Climate · geoengineering",
    provenance: "novel — speculative",
    body:
      "I propose a distributed ocean-surface albedo modulation system based on engineered cyanobacteria rafts with genetically tunable pigmentation. Each raft consists of a buoyant biocompatible scaffold colonized by photosynthetic cyanobacteria whose carotenoid and phycobiliprotein expression can be regulated by ambient light intensity together with a small-molecule inducer delivered through seawater. By switching their dominant pigment composition, individual rafts can shift between strongly absorbing and strongly reflective optical states on a timescale of hours. The rafts are not centrally controlled; instead they self-organize their spatial coverage through stigmergic chemical signaling inspired by slime mold aggregation, releasing and sensing volatile organic markers that modulate local growth and pigmentation in neighboring rafts. The collective behavior produces decentralized, reversible solar reflection patches that can expand during heat anomalies and dissipate when conditions normalize, offering a controllable and biodegradable alternative to stratospheric aerosol injection for regional climate intervention.",
  },
];

export default function IdeaInput({ onSubmit, disabled }: Props) {
  const [idea, setIdea] = useState("");
  const [evidenceSelection, setEvidenceSelection] =
    useState<EvidenceSelection | null>(null);
  const [focused, setFocused] = useState(false);
  const [activeSpecimen, setActiveSpecimen] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const charCount = idea.length;
  const isValid = charCount >= 50 && evidenceSelection !== null;

  const handleSourcesChange = useCallback((sel: EvidenceSelection | null) => {
    setEvidenceSelection(sel);
  }, []);

  const handleLoadSpecimen = useCallback((s: Specimen) => {
    setIdea(s.body);
    setActiveSpecimen(s.numeral);
    requestAnimationFrame(() => {
      const ta = textareaRef.current;
      if (!ta) return;
      ta.scrollIntoView({ behavior: "smooth", block: "center" });
      ta.focus({ preventScroll: true });
      ta.setSelectionRange(s.body.length, s.body.length);
    });
  }, []);

  return (
    <div className="grid grid-cols-12 gap-x-6 gap-y-12">
      {/* Marginalia */}
      <aside className="hidden lg:block col-span-2 pt-2">
        <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
          § i
        </p>
        <p className="font-display italic text-[color:var(--color-ink-soft)] leading-snug">
          The submission.
        </p>
        <div className="hairline mt-4 mb-4" />
        <p className="font-mono text-[10px] leading-relaxed text-[color:var(--color-ink-fade)]">
          State the hypothesis<br />
          in plain prose.<br />
          Specificity is novelty's<br />
          first witness.
        </p>
      </aside>

      {/* Manuscript page */}
      <motion.article
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.15 }}
        className="col-span-12 lg:col-span-10 relative"
      >
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight text-[color:var(--color-ink)]">
            On the Matter of Your Hypothesis
          </h2>
          <span className="small-caps text-[color:var(--color-ink-fade)] hidden md:inline">
            chapter one
          </span>
        </div>
        <p className="font-display italic text-base text-[color:var(--color-ink-fade)] mb-6 max-w-2xl">
          Describe the work as you would in a grant abstract · the problem, the
          method, the contribution. Hypothetica reads carefully.
        </p>

        <div
          className={`relative bg-[color:var(--color-paper-shade)] border border-[color:var(--color-rule)] transition-shadow ${
            focused ? "shadow-[0_2px_0_0_var(--color-vermillion)]" : ""
          }`}
        >
          {/* corner ornaments */}
          <span className="absolute top-2 left-3 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
            ms. i
          </span>
          <span className="absolute top-2 right-3 font-mono text-[10px] text-[color:var(--color-ink-fade)]">
            ✦
          </span>

          <textarea
            ref={textareaRef}
            value={idea}
            onChange={(e) => {
              setIdea(e.target.value);
              if (activeSpecimen) setActiveSpecimen(null);
            }}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder="e.g. I propose a multimodal retrieval-augmented generation system that reasons jointly over text documents and accompanying figures, using a shared embedding space trained with contrastive alignment…"
            rows={8}
            disabled={disabled}
            spellCheck
            className="block w-full bg-transparent px-8 pt-9 pb-10 font-body text-[15px] leading-relaxed text-[color:var(--color-ink)] placeholder:text-[color:var(--color-ink-mute)] placeholder:italic resize-none focus:outline-none disabled:opacity-50"
          />
        </div>

        {/* Page meta + submit */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mt-4">
          <div className="flex items-center gap-4 font-mono text-[11px] text-[color:var(--color-ink-fade)]">
            <span
              className={
                charCount >= 50
                  ? "text-[color:var(--color-moss)]"
                  : ""
              }
            >
              [{charCount.toString().padStart(4, "0")} / 0050 char min]
            </span>
            <span className="hidden sm:inline opacity-70">·</span>
            <span className="hidden sm:inline opacity-70">
              {idea.trim().split(/\s+/).filter(Boolean).length} words
            </span>
          </div>

          <button
            onClick={() =>
              evidenceSelection && onSubmit(idea, evidenceSelection)
            }
            disabled={disabled || !isValid}
            className="group relative inline-flex items-center gap-3 px-7 py-3.5 bg-[color:var(--color-ink)] text-[color:var(--color-paper)] font-display text-lg tracking-tight shadow-[0_5px_0_0_var(--color-vermillion)] hover:shadow-[0_2px_0_0_var(--color-vermillion)] hover:translate-y-[3px] disabled:opacity-30 disabled:pointer-events-none disabled:shadow-none disabled:translate-y-0 transition-all"
          >
            <span className="font-mono text-sm opacity-70 group-hover:translate-x-0.5 transition-transform">
              →
            </span>
            <span>Submit</span>
          </button>
        </div>

        {/* ─── Specimens from the Archive ──────────────────────── */}
        <div className="mt-10">
          <div className="hairline mb-6" />
          <div className="flex items-baseline justify-between mb-2">
            <h3 className="font-display text-xl tracking-tight text-[color:var(--color-ink)]">
              Specimens from the Archive
            </h3>
            <span className="small-caps text-[color:var(--color-ink-fade)] font-mono text-[11px]">
              vi exempla · click to load
            </span>
          </div>
          <p className="font-display italic text-sm text-[color:var(--color-ink-fade)] mb-5 max-w-2xl">
            Four canonical hypotheses · two speculative inventions. Provided
            for instrument calibration — click any to populate the manuscript above.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {SPECIMENS.map((s, i) => {
              const isActive = activeSpecimen === s.numeral;
              return (
                <motion.button
                  key={s.numeral}
                  type="button"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.45, delay: 0.25 + i * 0.06 }}
                  onClick={() => handleLoadSpecimen(s)}
                  disabled={disabled}
                  className={`group relative text-left bg-[color:var(--color-paper-shade)] border border-[color:var(--color-rule)] px-5 pt-4 pb-9 transition-all hover:translate-y-[-2px] hover:border-[color:var(--color-rule-strong)] hover:shadow-[0_4px_0_0_var(--color-vermillion)] disabled:opacity-40 disabled:pointer-events-none ${
                    isActive
                      ? "border-[color:var(--color-ink)] shadow-[0_4px_0_0_var(--color-vermillion)]"
                      : ""
                  }`}
                >
                  <div className="flex items-baseline justify-between mb-2">
                    <span className="font-mono text-[10px] text-[color:var(--color-ink-fade)] tracking-widest">
                      § {s.numeral}
                    </span>
                    <span className="small-caps text-[10px] text-[color:var(--color-ink-fade)]">
                      {s.domain}
                    </span>
                  </div>
                  <h4 className="font-display text-[17px] leading-snug text-[color:var(--color-ink)] mb-1.5">
                    {s.title}
                  </h4>
                  <p className="font-body italic text-[13px] leading-snug text-[color:var(--color-ink-soft)] line-clamp-3">
                    {s.body}
                  </p>
                  <div className="absolute bottom-2 left-5 right-5 flex items-center justify-between font-mono text-[10px] text-[color:var(--color-ink-fade)]">
                    <span className="opacity-80">{s.provenance}</span>
                    <span className="text-[color:var(--color-vermillion)] opacity-0 group-hover:opacity-100 transition-opacity">
                      load ↗
                    </span>
                  </div>
                  {isActive && (
                    <span className="absolute top-2 right-2 font-mono text-[10px] text-[color:var(--color-vermillion)]">
                      ✦ loaded
                    </span>
                  )}
                </motion.button>
              );
            })}
          </div>
        </div>
      </motion.article>

      {/* Source ribbon section */}
      <aside className="hidden lg:block col-span-2 pt-2">
        <p className="small-caps text-[color:var(--color-ink-fade)] mb-2">
          § ii
        </p>
        <p className="font-display italic text-[color:var(--color-ink-soft)] leading-snug">
          The corpus.
        </p>
        <div className="hairline mt-4 mb-4" />
        <p className="font-mono text-[10px] leading-relaxed text-[color:var(--color-ink-fade)]">
          Choose the archive<br />
          against which your idea<br />
          shall be cross-examined.
        </p>
      </aside>

      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.35 }}
        className="col-span-12 lg:col-span-10"
      >
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-display text-3xl md:text-4xl tracking-tight text-[color:var(--color-ink)]">
            Choose Your Tribunal
          </h2>
          <span className="small-caps text-[color:var(--color-ink-fade)] hidden md:inline">
            chapter two
          </span>
        </div>
        <p className="font-display italic text-base text-[color:var(--color-ink-fade)] mb-6 max-w-2xl">
          Each archive surfaces different evidence · peer-reviewed papers, prior
          art in patents, or open-source artefacts.
        </p>
        <SourceSelection
          onSourcesChange={handleSourcesChange}
          disabled={disabled}
        />
      </motion.div>
    </div>
  );
}
