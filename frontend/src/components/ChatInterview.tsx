import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { ChatMessage } from "../types/api";

interface Props {
  messages: ChatMessage[];
  isAiTyping: boolean;
  currentRound: number;
  maxRounds: number;
  onSendMessage: (message: string) => void;
  onSkip: () => void;
  onEndEarly: () => void;
}

const CATEGORY: Record<string, { label: string; accent: string }> = {
  problem: { label: "Problem", accent: "var(--color-vermillion)" },
  method: { label: "Method", accent: "var(--color-moss)" },
  novelty: { label: "Novelty", accent: "var(--color-ochre)" },
  application: { label: "Application", accent: "var(--color-gold)" },
};

function Typewriter({ text }: { text: string }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (reduced) {
      const t = window.setTimeout(() => setShown(text.length), 0);
      return () => window.clearTimeout(t);
    }
    const tReset = window.setTimeout(() => setShown(0), 0);
    let i = 0;
    const total = text.length;
    const perChar = Math.max(8, Math.min(28, 900 / total));
    const id = window.setInterval(() => {
      i += 1;
      setShown(i);
      if (i >= total) window.clearInterval(id);
    }, perChar);
    return () => {
      window.clearTimeout(tReset);
      window.clearInterval(id);
    };
  }, [text]);
  const done = shown >= text.length;
  return (
    <span className={done ? "" : "caret-blink"}>{text.slice(0, shown)}</span>
  );
}

export default function ChatInterview({
  messages,
  isAiTyping,
  currentRound,
  maxRounds,
  onSendMessage,
  onSkip,
  onEndEarly,
}: Props) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const hasUserMessages = messages.some((m) => m.role === "user");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isAiTyping]);

  useEffect(() => {
    if (!isAiTyping) textareaRef.current?.focus();
  }, [isAiTyping]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isAiTyping) return;
    onSendMessage(text);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Marginalia */}
      <aside className="hidden lg:flex col-span-3 flex-col gap-4 pt-2">
        <p className="small-caps text-[color:var(--color-ink-fade)]">
          § examination
        </p>
        <p className="font-display italic text-xl text-[color:var(--color-ink-soft)] leading-tight">
          “We must understand the hypothesis before we assess it.”
        </p>
        <div className="hairline" />
        <div className="font-mono text-[11px] text-[color:var(--color-ink-fade)] space-y-1">
          <div className="flex items-center justify-between">
            <span>round</span>
            <span className="numeric-tabular">
              {currentRound.toString().padStart(2, "0")} / {maxRounds.toString().padStart(2, "0")}
            </span>
          </div>
          <div className="flex gap-1 pt-1">
            {Array.from({ length: maxRounds }, (_, i) => (
              <span
                key={i}
                className={`h-px flex-1 transition-all duration-500 ${
                  i < currentRound
                    ? "bg-[color:var(--color-vermillion)]"
                    : "bg-[color:var(--color-rule)]"
                }`}
                style={{ height: i < currentRound ? "3px" : "1px" }}
              />
            ))}
          </div>
        </div>
        <div className="hairline" />
        <div className="flex flex-col gap-2">
          {!hasUserMessages ? (
            <button
              onClick={onSkip}
              className="group relative inline-flex items-center gap-2 self-start px-4 py-2.5 border border-[color:var(--color-rule-strong)] bg-[color:var(--color-paper)] text-[color:var(--color-ink)] small-caps font-bold hover:bg-[color:var(--color-ink)] hover:text-[color:var(--color-paper)] hover:border-[color:var(--color-ink)] transition-colors"
            >
              <span className="transition-transform group-hover:-translate-x-0.5">↷</span>
              <span>skip examination</span>
            </button>
          ) : (
            <button
              onClick={onEndEarly}
              className="group relative inline-flex items-center gap-2 self-start px-4 py-2.5 bg-[color:var(--color-vermillion)] text-[color:var(--color-paper)] small-caps font-bold shadow-[0_4px_0_0_var(--color-ink)] hover:shadow-[0_2px_0_0_var(--color-ink)] hover:translate-y-0.5 transition-all"
            >
              <span className="transition-transform group-hover:rotate-12">✦</span>
              <span>assess now</span>
              <span className="font-mono opacity-70">→</span>
            </button>
          )}
        </div>
      </aside>

      {/* Chat column */}
      <div className="col-span-12 lg:col-span-9 relative">
        <div className="flex items-baseline justify-between mb-2 lg:hidden">
          <h2 className="font-display text-3xl tracking-tight">
            Examination
          </h2>
          <span className="font-mono text-xs text-[color:var(--color-ink-fade)]">
            round {currentRound}/{maxRounds}
          </span>
        </div>

        <h2 className="hidden lg:block font-display text-4xl tracking-tight mb-2">
          On Examination
        </h2>
        <p className="font-display italic text-[color:var(--color-ink-fade)] mb-6">
          The board will ask after problem, method, novelty, and application.
        </p>

        <div className="relative bg-[color:var(--color-paper-shade)] border border-[color:var(--color-rule)]">
          <div
            className="h-[480px] overflow-y-auto px-6 py-6 space-y-6"
            aria-live="polite"
          >
            <AnimatePresence initial={false}>
              {messages.map((msg) => {
                const cat = msg.category ? CATEGORY[msg.category] : undefined;
                if (msg.role === "ai") {
                  return (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4 }}
                      className="max-w-2xl"
                    >
                      <div className="flex items-center gap-3 mb-1.5">
                        <span className="small-caps text-[color:var(--color-ink-fade)]">
                          the board asks
                        </span>
                        {cat && (
                          <span
                            className="small-caps px-2 py-0.5"
                            style={{
                              color: cat.accent,
                              borderBottom: `1px solid ${cat.accent}`,
                            }}
                          >
                            {cat.label}
                          </span>
                        )}
                      </div>
                      <p className="font-display text-xl md:text-[1.4rem] leading-snug text-[color:var(--color-ink)]">
                        <Typewriter text={msg.content} />
                      </p>
                    </motion.div>
                  );
                }
                return (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, scale: 1.05, rotate: -1 }}
                    animate={{ opacity: 1, scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 220, damping: 16 }}
                    className="ml-auto max-w-2xl text-right"
                  >
                    <p className="small-caps text-[color:var(--color-vermillion)] mb-1.5">
                      the author replies
                    </p>
                    <div className="inline-block text-left border-l-2 border-[color:var(--color-vermillion)] pl-4 pr-1 py-1 bg-[color:var(--color-paper)]">
                      <p className="font-body text-[15px] leading-relaxed text-[color:var(--color-ink)]">
                        {msg.content}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {isAiTyping && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2 text-[color:var(--color-ink-fade)]"
              >
                <span className="small-caps">the board deliberates</span>
                <span className="flex gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[color:var(--color-ink-fade)] pulse-dot" />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-[color:var(--color-ink-fade)] pulse-dot"
                    style={{ animationDelay: "0.2s" }}
                  />
                  <span
                    className="w-1.5 h-1.5 rounded-full bg-[color:var(--color-ink-fade)] pulse-dot"
                    style={{ animationDelay: "0.4s" }}
                  />
                </span>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-[color:var(--color-rule)] px-6 py-4 bg-[color:var(--color-paper)]">
            <div className="flex items-start gap-3">
              <span className="font-mono text-xs text-[color:var(--color-ink-fade)] pt-3 select-none">
                ›
              </span>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                disabled={isAiTyping}
                rows={1}
                placeholder={
                  isAiTyping
                    ? "the board is deliberating…"
                    : "compose your reply…"
                }
                className="flex-1 resize-none bg-transparent font-body text-[15px] leading-relaxed text-[color:var(--color-ink)] placeholder:italic placeholder:text-[color:var(--color-ink-mute)] focus:outline-none disabled:opacity-50"
                style={{ minHeight: "40px", maxHeight: "140px" }}
              />
              <button
                onClick={handleSend}
                disabled={isAiTyping || !input.trim()}
                className="group self-end inline-flex items-center gap-2 px-4 py-2 bg-[color:var(--color-ink)] text-[color:var(--color-paper)] small-caps font-bold disabled:opacity-25 disabled:pointer-events-none hover:bg-[color:var(--color-vermillion)] transition-colors"
              >
                <span>send</span>
                <span className="font-mono group-hover:translate-x-0.5 transition-transform">↵</span>
              </button>
            </div>
            <p className="mt-2 font-mono text-[10px] text-[color:var(--color-ink-fade)] text-right">
              ↵ to send · ⇧↵ for new line
            </p>
          </div>
        </div>

        {/* Mobile actions */}
        <div className="flex lg:hidden mt-4 gap-3">
          {!hasUserMessages ? (
            <button
              onClick={onSkip}
              className="px-4 py-2 border border-[color:var(--color-rule-strong)] small-caps font-bold text-[color:var(--color-ink)] hover:bg-[color:var(--color-ink)] hover:text-[color:var(--color-paper)] transition-colors"
            >
              ↷ skip
            </button>
          ) : (
            <button
              onClick={onEndEarly}
              className="px-4 py-2 bg-[color:var(--color-vermillion)] text-[color:var(--color-paper)] small-caps font-bold shadow-[0_4px_0_0_var(--color-ink)] hover:translate-y-0.5 hover:shadow-[0_2px_0_0_var(--color-ink)] transition-all"
            >
              ✦ assess now →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
