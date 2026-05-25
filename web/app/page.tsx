"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { QueryClassifier, type ClassifierStatus } from "@/lib/classifier";
import {
  buildSearchUrls,
  explainHeuristics,
  resultFromHeuristics,
  type ClassificationResult,
  type HeuristicExplanation,
  type InferenceSource,
} from "@/lib/heuristics";

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

function sourceLabel(source: InferenceSource): string {
  return source === "neural" ? "Neural network" : "Heuristic fallback";
}

export default function Home() {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 200);
  const [status, setStatus] = useState<ClassifierStatus>("idle");
  const [modelError, setModelError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [heuristics, setHeuristics] = useState<HeuristicExplanation | null>(null);
  const [feedbackSent, setFeedbackSent] = useState<"up" | "down" | null>(null);
  const classifierRef = useRef<QueryClassifier | null>(null);
  const modelReadyRef = useRef(false);

  useEffect(() => {
    classifierRef.current = new QueryClassifier(setStatus);
    classifierRef.current
      .init()
      .then(() => {
        modelReadyRef.current = true;
      })
      .catch((err) => {
        console.error(err);
        setModelError(err instanceof Error ? err.message : String(err));
      });
    return () => classifierRef.current?.destroy();
  }, []);

  const classify = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResult(null);
      setHeuristics(null);
      return;
    }

    setHeuristics(explainHeuristics(q));
    setFeedbackSent(null);

    if (modelError || (!modelReadyRef.current && status === "error")) {
      setResult(resultFromHeuristics(q));
      return;
    }

    try {
      const res = await classifierRef.current!.classify(q);
      setResult(res);
    } catch (err) {
      console.error("Classification failed:", err);
      setResult(resultFromHeuristics(q));
    }
  }, [status, modelError]);

  useEffect(() => {
    classify(debouncedQuery);
  }, [debouncedQuery, classify]);

  const sendFeedback = async (correct: boolean) => {
    if (!result || !query.trim()) return;
    setFeedbackSent(correct ? "up" : "down");

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          predicted_label: result.label,
          confidence: result.confidence,
          source: result.source,
          correct,
        }),
      });
    } catch {
      // Non-critical
    }
  };

  const urls = query.trim() ? buildSearchUrls(query) : null;
  const pct = result ? Math.round(result.confidence * 100) : 0;
  const usingNeural = result?.source === "neural";

  return (
    <main>
      <header className="hero">
        <h1 className="wordmark">
          <span className="wordmark-cn" lang="zh-Hans">
            解惑
          </span>
          <span className="wordmark-en">JieHuo</span>
        </h1>
        <p className="subtitle">
          Multilingual query router — instantly decides whether your search belongs on{" "}
          <strong>Google</strong> or <strong>Perplexity</strong>. Runs entirely in your browser.
        </p>

        <div className="status-bar">
          <span className={`status-dot ${status}`} />
          {status === "loading" && "Loading model (~130 MB from Hugging Face)…"}
          {status === "ready" && "Neural model ready"}
          {status === "idle" && "Initializing…"}
          {status === "error" && "Model failed — routing uses heuristics only"}
        </div>
      </header>

      <div className="search-box">
        <input
          className="search-input"
          type="text"
          placeholder="Ask anything, in any language…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
      </div>

      {!query.trim() ? (
        <div className="result-card empty">
          <p>Try &ldquo;weather in Tokyo today&rdquo; or &ldquo;为什么人工智能会改变医疗&rdquo;</p>
        </div>
      ) : (
        <div className="result-card">
          {result ? (
            <>
              <div className="badge-row">
                <span className={`source-badge ${result.source}`}>
                  Source: {sourceLabel(result.source)}
                </span>
                <span className={`route-badge ${result.label}`}>
                  Route to {result.label === "google" ? "Google" : "Perplexity"}
                  <span>· {pct}%</span>
                </span>
              </div>

              <div className="confidence-bar">
                <div
                  className={`confidence-fill ${result.label}`}
                  style={{ width: `${pct}%` }}
                />
              </div>

              <div className="scores">
                <span>Google: {(result.scores.google * 100).toFixed(1)}%</span>
                <span>Perplexity: {(result.scores.perplexity * 100).toFixed(1)}%</span>
              </div>

              {result.rawLabels && result.rawLabels.length > 0 && (
                <p className="raw-labels">
                  Model output: {result.rawLabels.join(", ")}
                </p>
              )}
            </>
          ) : (
            <p className="classifying">Classifying…</p>
          )}

          {heuristics && (
            <div className="explain">
              <div className="explain-title">
                Rule-based signals {usingNeural ? "(reference only — not used for routing)" : "(used for routing)"}
              </div>
              <div className="explain-tags">
                {heuristics.reasons.map((r) => (
                  <span key={r} className="tag">
                    {r}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="actions">
            <a
              className={`btn btn-google${result?.label === "google" ? " recommended" : ""}`}
              href={urls?.google}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open in Google
              <span className="btn-arrow" aria-hidden="true">↗</span>
            </a>
            <a
              className={`btn btn-perplexity${result?.label === "perplexity" ? " recommended" : ""}`}
              href={urls?.perplexity}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open in Perplexity
              <span className="btn-arrow" aria-hidden="true">↗</span>
            </a>
          </div>

          {result && (
            <div className="feedback">
              <span>Was this routing correct?</span>
              <button
                className={`feedback-btn ${feedbackSent === "up" ? "active" : ""}`}
                onClick={() => sendFeedback(true)}
                disabled={feedbackSent !== null}
                aria-label="Yes, routing was correct"
              >
                👍
              </button>
              <button
                className={`feedback-btn ${feedbackSent === "down" ? "active" : ""}`}
                onClick={() => sendFeedback(false)}
                disabled={feedbackSent !== null}
                aria-label="No, routing was incorrect"
              >
                👎
              </button>
              {feedbackSent && <span className="feedback-thanks">Thanks!</span>}
            </div>
          )}
        </div>
      )}

      <footer className="footer">
        Model: KenWu/multilingual-query-router · trained on 518 synthetic queries · e5-small
      </footer>
    </main>
  );
}
