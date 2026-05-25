import type { ClassificationResult } from "./heuristics";

type WorkerRequest =
  | { type: "init" }
  | { type: "classify"; id: number; query: string };

type WorkerResponse =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "result"; id: number; result: ClassificationResult };

export type ClassifierStatus = "idle" | "loading" | "ready" | "error";

export class QueryClassifier {
  private worker: Worker | null = null;
  private status: ClassifierStatus = "idle";
  private pending = new Map<number, (result: ClassificationResult) => void>();
  private nextId = 0;
  private initPromise: Promise<void> | null = null;
  private onStatusChange?: (status: ClassifierStatus) => void;

  constructor(onStatusChange?: (status: ClassifierStatus) => void) {
    this.onStatusChange = onStatusChange;
  }

  getStatus(): ClassifierStatus {
    return this.status;
  }

  private setStatus(status: ClassifierStatus) {
    this.status = status;
    this.onStatusChange?.(status);
  }

  async init(): Promise<void> {
    if (this.status === "ready") return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      this.setStatus("loading");

      this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });

      const onMessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data;
        if (msg.type === "ready") {
          this.setStatus("ready");
          this.worker?.removeEventListener("message", onMessage);
          resolve();
        } else if (msg.type === "error") {
          this.setStatus("error");
          this.worker?.removeEventListener("message", onMessage);
          reject(new Error(msg.message));
        }
      };

      this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data;
        if (msg.type === "result") {
          const cb = this.pending.get(msg.id);
          if (cb) {
            this.pending.delete(msg.id);
            cb(msg.result);
          }
        } else {
          onMessage(event);
        }
      });

      this.worker.postMessage({ type: "init" } satisfies WorkerRequest);
    });

    return this.initPromise;
  }

  async classify(query: string): Promise<ClassificationResult> {
    await this.init();
    if (!this.worker) throw new Error("Worker not initialized");

    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.worker!.postMessage({ type: "classify", id, query } satisfies WorkerRequest);
    });
  }

  destroy() {
    this.worker?.terminate();
    this.worker = null;
    this.setStatus("idle");
    this.initPromise = null;
  }
}
