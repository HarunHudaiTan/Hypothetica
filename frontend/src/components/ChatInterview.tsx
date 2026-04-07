import { useState, useRef, useEffect } from "react";
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

const categoryConfig: Record<string, { label: string; color: string }> = {
  problem: { label: "Problem", color: "bg-rose-100 text-rose-700" },
  method: { label: "Method", color: "bg-sky-100 text-sky-700" },
  novelty: { label: "Novelty", color: "bg-amber-100 text-amber-700" },
  application: { label: "Application", color: "bg-emerald-100 text-emerald-700" },
};

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
    if (!isAiTyping) {
      textareaRef.current?.focus();
    }
  }, [isAiTyping]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isAiTyping) return;
    onSendMessage(text);
    setInput("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-expand textarea
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col" style={{ height: "600px" }}>
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">Research Interview</h2>
            <p className="text-sm text-slate-500 mt-0.5">
              Let's understand your idea better
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Round indicator */}
            <div className="flex items-center gap-1.5">
              {Array.from({ length: maxRounds }, (_, i) => (
                <div
                  key={i}
                  className={`h-1.5 w-6 rounded-full transition-all duration-300 ${
                    i < currentRound
                      ? "bg-indigo-500"
                      : "bg-slate-200"
                  }`}
                />
              ))}
              <span className="text-xs text-slate-400 ml-1.5">
                {currentRound}/{maxRounds}
              </span>
            </div>

            {/* Action buttons */}
            {!hasUserMessages ? (
              <button
                onClick={onSkip}
                className="text-xs text-slate-400 hover:text-slate-600 px-2 py-1 rounded transition-colors"
              >
                Skip interview
              </button>
            ) : (
              <button
                onClick={onEndEarly}
                className="text-xs text-indigo-500 hover:text-indigo-700 font-medium px-2 py-1 rounded transition-colors"
              >
                Analyze now
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-[fadeSlideIn_0.3s_ease-out]`}
            style={{ animationDelay: `${idx === messages.length - 1 ? 0 : 0}ms` }}
          >
            {msg.role === "ai" ? (
              <div className="max-w-[85%]">
                {msg.category && categoryConfig[msg.category] && (
                  <span
                    className={`inline-block text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full mb-1.5 ${categoryConfig[msg.category].color}`}
                  >
                    {categoryConfig[msg.category].label}
                  </span>
                )}
                <div className="bg-slate-50 border border-slate-150 rounded-2xl rounded-tl-sm px-4 py-3">
                  <p className="text-sm text-slate-800 leading-relaxed">{msg.content}</p>
                </div>
              </div>
            ) : (
              <div className="max-w-[85%]">
                <div className="bg-indigo-600 rounded-2xl rounded-tr-sm px-4 py-3">
                  <p className="text-sm text-white leading-relaxed">{msg.content}</p>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {isAiTyping && (
          <div className="flex justify-start animate-[fadeSlideIn_0.3s_ease-out]">
            <div className="bg-slate-50 border border-slate-150 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-[dotBounce_1.4s_ease-in-out_infinite]" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-[dotBounce_1.4s_ease-in-out_0.2s_infinite]" />
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-[dotBounce_1.4s_ease-in-out_0.4s_infinite]" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="flex-none px-6 py-4 border-t border-slate-100 bg-slate-50/50">
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={isAiTyping}
            rows={1}
            placeholder={isAiTyping ? "Waiting for question..." : "Type your answer..."}
            className="flex-1 resize-none rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/10 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            style={{ minHeight: "40px", maxHeight: "120px" }}
          />
          <button
            onClick={handleSend}
            disabled={isAiTyping || !input.trim()}
            className="flex-none w-10 h-10 flex items-center justify-center rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </button>
        </div>
        <p className="text-[11px] text-slate-400 mt-2 text-center">
          Press Enter to send &middot; Shift+Enter for new line
        </p>
      </div>

      {/* Inline animations */}
      <style>{`
        @keyframes fadeSlideIn {
          from {
            opacity: 0;
            transform: translateY(8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes dotBounce {
          0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.4;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
