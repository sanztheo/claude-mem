/**
 * CodexAgent: OpenAI Codex CLI-based observation extraction
 *
 * Alternative to SDKAgent that uses the OpenAI Codex CLI (`codex exec --json`)
 * for accessing GPT-5-Codex family models (gpt-5-codex, gpt-5.2-codex, gpt-5.4-codex)
 * via ChatGPT subscription (codex login) or OPENAI_API_KEY.
 *
 * Responsibility:
 * - Spawn `codex exec --json` per LLM turn
 * - Parse the JSONL event stream (thread.started, turn.*, item.*)
 * - Strip markdown fences from responses (local fix for #1874)
 * - Sync to database and Chroma via shared ResponseProcessor
 * - Support dynamic model selection across Codex variants
 */

import { spawn } from "child_process";
import { tmpdir } from "os";
import { getCredential } from "../../shared/EnvManager.js";
import { SettingsDefaultsManager } from "../../shared/SettingsDefaultsManager.js";
import { USER_SETTINGS_PATH } from "../../shared/paths.js";
import { logger } from "../../utils/logger.js";
import {
  buildContinuationPrompt,
  buildInitPrompt,
  buildObservationPrompt,
  buildSummaryPrompt,
} from "../../sdk/prompts.js";
import { ModeManager } from "../domain/ModeManager.js";
import type { ModeConfig } from "../domain/types.js";
import type { ActiveSession, ConversationMessage } from "../worker-types.js";
import { DatabaseManager } from "./DatabaseManager.js";
import { SessionManager } from "./SessionManager.js";
import {
  isAbortError,
  processAgentResponse,
  shouldFallbackToClaude,
  type FallbackAgent,
  type WorkerRef,
} from "./agents/index.js";

// Default context window limits (overridable via settings)
const DEFAULT_MAX_CONTEXT_MESSAGES = 20;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

// Hard timeout for a single codex exec invocation
const CODEX_EXEC_TIMEOUT_MS = 120_000;

// Default model — latest Codex variant at the time of writing
const DEFAULT_CODEX_MODEL = "gpt-5.4-codex";
const DEFAULT_CODEX_REASONING = "medium";

interface CodexResult {
  content: string;
  tokensUsed?: number;
  threadId?: string;
}

interface CodexUsage {
  input_tokens?: number;
  cached_input_tokens?: number;
  output_tokens?: number;
}

interface CodexEvent {
  type: string;
  thread_id?: string;
  item?: {
    id?: string;
    type?: string;
    text?: string;
  };
  usage?: CodexUsage;
  error?: { message?: string };
  message?: string;
}

export class CodexAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set fallback agent (Claude SDK) for when Codex CLI fails.
   * Must be set after construction to avoid circular dependency.
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   * Start Codex agent for a session.
   * Each LLM turn is a fresh `codex exec` call — the conversation history is
   * stored in session.conversationHistory and re-sent as a concatenated prompt
   * (Codex exec is stateless per invocation, same approach as OpenRouter).
   */
  async startSession(
    session: ActiveSession,
    worker?: WorkerRef,
  ): Promise<void> {
    const config = this.getCodexConfig();
    await this.ensureCodexAvailable(config);

    // Synthetic memorySessionId — Codex exec resume exists but is not required for MVP
    // (see SettingsDefaultsManager: CLAUDE_MEM_CODEX_USE_RESUME reserved for v2)
    if (!session.memorySessionId) {
      const syntheticId = `codex-${session.contentSessionId}-${Date.now()}`;
      session.memorySessionId = syntheticId;
      this.dbManager
        .getSessionStore()
        .updateMemorySessionId(session.sessionDbId, syntheticId);
      logger.info(
        "SESSION",
        `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=Codex`,
      );
    }

    const mode = ModeManager.getInstance().getActiveMode();

    const initPrompt =
      session.lastPromptNumber === 1
        ? buildInitPrompt(
            session.project,
            session.contentSessionId,
            session.userPrompt,
            mode,
          )
        : buildContinuationPrompt(
            session.userPrompt,
            session.lastPromptNumber,
            session.contentSessionId,
            mode,
          );

    session.conversationHistory.push({ role: "user", content: initPrompt });

    try {
      const initResponse = await this.queryCodex(
        session.conversationHistory,
        config,
      );
      await this.handleInitResponse(
        initResponse,
        session,
        worker,
        config.model,
      );
    } catch (error: unknown) {
      this.logAgentError("Codex init failed", session, config.model, error);
      await this.handleSessionError(error, session, worker);
      return;
    }

    let lastCwd: string | undefined;

    try {
      for await (const message of this.sessionManager.getMessageIterator(
        session.sessionDbId,
      )) {
        lastCwd = await this.processOneMessage(
          session,
          message,
          lastCwd,
          config,
          worker,
          mode,
        );
      }
    } catch (error: unknown) {
      this.logAgentError(
        "Codex message processing failed",
        session,
        config.model,
        error,
      );
      await this.handleSessionError(error, session, worker);
      return;
    }

    const durationMs = Date.now() - session.startTime;
    logger.success("SDK", "Codex agent completed", {
      sessionId: session.sessionDbId,
      duration: `${(durationMs / 1000).toFixed(1)}s`,
      historyLength: session.conversationHistory.length,
      model: config.model,
    });
  }

  private prepareMessageMetadata(
    session: ActiveSession,
    message: {
      _persistentId: number;
      agentId?: string | null;
      agentType?: string | null;
    },
  ): void {
    session.processingMessageIds.push(message._persistentId);
    session.pendingAgentId = message.agentId ?? null;
    session.pendingAgentType = message.agentType ?? null;
  }

  private async handleInitResponse(
    initResponse: CodexResult,
    session: ActiveSession,
    worker: WorkerRef | undefined,
    model: string,
  ): Promise<void> {
    if (initResponse.content) {
      session.conversationHistory.push({
        role: "assistant",
        content: initResponse.content,
      });
      const tokensUsed = initResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

      await processAgentResponse(
        initResponse.content,
        session,
        this.dbManager,
        this.sessionManager,
        worker,
        tokensUsed,
        null,
        "Codex",
        undefined,
        model,
      );
    } else {
      logger.error(
        "SDK",
        "Empty Codex init response — session may lack context",
        {
          sessionId: session.sessionDbId,
          model,
        },
      );
    }
  }

  private async processOneMessage(
    session: ActiveSession,
    message: {
      _persistentId: number;
      agentId?: string | null;
      agentType?: string | null;
      type?: string;
      cwd?: string;
      prompt_number?: number;
      tool_name?: string;
      tool_input?: unknown;
      tool_response?: unknown;
      last_assistant_message?: string;
    },
    lastCwd: string | undefined,
    config: CodexConfig,
    worker: WorkerRef | undefined,
    mode: ModeConfig,
  ): Promise<string | undefined> {
    this.prepareMessageMetadata(session, message);

    if (message.cwd) {
      lastCwd = message.cwd;
    }
    const originalTimestamp = session.earliestPendingTimestamp;

    if (message.type === "observation") {
      await this.processObservationMessage(
        session,
        message,
        originalTimestamp,
        lastCwd,
        config,
        worker,
      );
    } else if (message.type === "summarize") {
      await this.processSummaryMessage(
        session,
        message,
        originalTimestamp,
        lastCwd,
        config,
        worker,
        mode,
      );
    }

    return lastCwd;
  }

  private async processObservationMessage(
    session: ActiveSession,
    message: {
      prompt_number?: number;
      tool_name?: string;
      tool_input?: unknown;
      tool_response?: unknown;
      cwd?: string;
    },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    config: CodexConfig,
    worker: WorkerRef | undefined,
  ): Promise<void> {
    if (message.prompt_number !== undefined) {
      session.lastPromptNumber = message.prompt_number;
    }

    if (!session.memorySessionId) {
      throw new Error(
        "Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.",
      );
    }

    const obsPrompt = buildObservationPrompt({
      id: 0,
      tool_name: message.tool_name!,
      tool_input: JSON.stringify(message.tool_input),
      tool_output: JSON.stringify(message.tool_response),
      created_at_epoch: originalTimestamp ?? Date.now(),
      cwd: message.cwd,
    });

    session.conversationHistory.push({ role: "user", content: obsPrompt });
    const obsResponse = await this.queryCodex(
      session.conversationHistory,
      config,
    );

    let tokensUsed = 0;
    if (obsResponse.content) {
      session.conversationHistory.push({
        role: "assistant",
        content: obsResponse.content,
      });
      tokensUsed = obsResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      obsResponse.content || "",
      session,
      this.dbManager,
      this.sessionManager,
      worker,
      tokensUsed,
      originalTimestamp,
      "Codex",
      lastCwd,
      config.model,
    );
  }

  private async processSummaryMessage(
    session: ActiveSession,
    message: { last_assistant_message?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    config: CodexConfig,
    worker: WorkerRef | undefined,
    mode: ModeConfig,
  ): Promise<void> {
    if (!session.memorySessionId) {
      throw new Error(
        "Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.",
      );
    }

    const summaryPrompt = buildSummaryPrompt(
      {
        id: session.sessionDbId,
        memory_session_id: session.memorySessionId,
        project: session.project,
        user_prompt: session.userPrompt,
        last_assistant_message: message.last_assistant_message || "",
      },
      mode,
    );

    session.conversationHistory.push({ role: "user", content: summaryPrompt });
    const summaryResponse = await this.queryCodex(
      session.conversationHistory,
      config,
    );

    let tokensUsed = 0;
    if (summaryResponse.content) {
      session.conversationHistory.push({
        role: "assistant",
        content: summaryResponse.content,
      });
      tokensUsed = summaryResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      summaryResponse.content || "",
      session,
      this.dbManager,
      this.sessionManager,
      worker,
      tokensUsed,
      originalTimestamp,
      "Codex",
      lastCwd,
      config.model,
    );
  }

  private async handleSessionError(
    error: unknown,
    session: ActiveSession,
    worker?: WorkerRef,
  ): Promise<never | void> {
    if (isAbortError(error)) {
      logger.warn("SDK", "Codex agent aborted", {
        sessionId: session.sessionDbId,
      });
      throw error;
    }

    if (shouldFallbackToClaude(error) && this.fallbackAgent) {
      logger.warn("SDK", "Codex CLI failed, falling back to Claude SDK", {
        sessionDbId: session.sessionDbId,
        error: error instanceof Error ? error.message : String(error),
        historyLength: session.conversationHistory.length,
      });
      await this.fallbackAgent.startSession(session, worker);
      return;
    }

    logger.failure(
      "SDK",
      "Codex agent error",
      { sessionDbId: session.sessionDbId },
      error instanceof Error ? error : new Error(String(error)),
    );
    throw error;
  }

  private logAgentError(
    label: string,
    session: ActiveSession,
    model: string,
    error: unknown,
  ): void {
    const wrapped = error instanceof Error ? error : new Error(String(error));
    logger.error(
      "SDK",
      label,
      { sessionId: session.sessionDbId, model },
      wrapped,
    );
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  private truncateHistory(
    history: ConversationMessage[],
  ): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const maxMessages =
      parseInt(settings.CLAUDE_MEM_CODEX_MAX_CONTEXT_MESSAGES) ||
      DEFAULT_MAX_CONTEXT_MESSAGES;
    const maxTokens =
      parseInt(settings.CLAUDE_MEM_CODEX_MAX_TOKENS) ||
      DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= maxMessages) {
      const totalTokens = history.reduce(
        (sum, m) => sum + this.estimateTokens(m.content),
        0,
      );
      if (totalTokens <= maxTokens) {
        return history;
      }
    }

    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (
        truncated.length >= maxMessages ||
        tokenCount + msgTokens > maxTokens
      ) {
        logger.warn(
          "SDK",
          "Codex context window truncated to prevent runaway costs",
          {
            originalMessages: history.length,
            keptMessages: truncated.length,
            droppedMessages: i + 1,
            estimatedTokens: tokenCount,
            tokenLimit: maxTokens,
          },
        );
        break;
      }

      truncated.unshift(msg);
      tokenCount += msgTokens;
    }

    return truncated;
  }

  /**
   * Convert conversation history to a single prompt string for `codex exec`.
   * Codex exec is stateless per invocation — we re-send the full turn context as text.
   * Format : blocks labelled `User:` / `Assistant:` separated by blank lines.
   */
  private conversationToPrompt(history: ConversationMessage[]): string {
    const parts: string[] = [];
    for (const msg of history) {
      const label = msg.role === "assistant" ? "Assistant" : "User";
      parts.push(`${label}:\n${msg.content}`);
    }
    // Trailing marker so Codex understands we want it to produce the next assistant turn
    parts.push("Assistant:");
    return parts.join("\n\n");
  }

  /**
   * Strip leading/trailing markdown code fences from Codex output.
   * Fix for #1874: LLMs (Claude, Codex, GPT) often wrap XML in ```xml ... ``` fences
   * which defeats the strict XML parser in ResponseProcessor.
   */
  private stripMarkdownFences(text: string): string {
    if (!text) return text;
    const trimmed = text.trim();
    // Match ```xml\n...\n``` or ```\n...\n``` (with optional language tag)
    const fenceMatch = trimmed.match(
      /^```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)\n```$/,
    );
    return fenceMatch ? fenceMatch[1].trim() : trimmed;
  }

  /**
   * Spawn `codex exec --json` with the concatenated conversation prompt.
   * Parses the JSONL event stream, returning the final agent_message text
   * and usage metrics. Throws on turn.failed or non-zero exit.
   */
  private async queryCodex(
    history: ConversationMessage[],
    config: CodexConfig,
  ): Promise<CodexResult> {
    const truncatedHistory = this.truncateHistory(history);
    const prompt = this.conversationToPrompt(truncatedHistory);

    logger.debug("SDK", `Querying Codex CLI (${config.model})`, {
      turns: truncatedHistory.length,
      totalChars: prompt.length,
      estimatedTokens: this.estimateTokens(prompt),
    });

    const args = [
      "exec",
      "--json",
      "--skip-git-repo-check",
      "--model",
      config.model,
      "--sandbox",
      "read-only",
      "-c",
      `model_reasoning_effort="${config.reasoning}"`,
      prompt,
    ];

    const env: NodeJS.ProcessEnv = { ...process.env };
    if (config.authMethod === "api" && config.apiKey) {
      env.OPENAI_API_KEY = config.apiKey;
    }

    return new Promise<CodexResult>((resolve, reject) => {
      const child = spawn(config.binary, args, {
        cwd: tmpdir(),
        env,
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdoutBuffer = "";
      let stderrBuffer = "";
      const agentMessageTexts: string[] = [];
      let threadId: string | undefined;
      let usage: CodexUsage | undefined;
      let failureMessage: string | undefined;

      const timeoutHandle = setTimeout(() => {
        child.kill("SIGKILL");
        reject(
          new Error(`Codex CLI timed out after ${CODEX_EXEC_TIMEOUT_MS}ms`),
        );
      }, CODEX_EXEC_TIMEOUT_MS);

      const processLine = (line: string): void => {
        const trimmed = line.trim();
        // Codex also emits stderr-like log lines to stdout occasionally — skip non-JSON
        if (!trimmed.startsWith("{")) return;

        let event: CodexEvent;
        try {
          event = JSON.parse(trimmed) as CodexEvent;
        } catch {
          return;
        }

        switch (event.type) {
          case "thread.started":
            threadId = event.thread_id;
            break;
          case "item.completed":
            if (event.item?.type === "agent_message" && event.item.text) {
              agentMessageTexts.push(event.item.text);
            }
            break;
          case "turn.completed":
            usage = event.usage;
            break;
          case "turn.failed":
            failureMessage =
              event.error?.message || "Codex turn failed without error message";
            break;
          case "error":
            failureMessage = event.message || "Codex emitted error event";
            break;
        }
      };

      child.stdout?.on("data", (chunk: Buffer) => {
        stdoutBuffer += chunk.toString("utf8");
        let newlineIdx: number;
        while ((newlineIdx = stdoutBuffer.indexOf("\n")) !== -1) {
          const line = stdoutBuffer.slice(0, newlineIdx);
          stdoutBuffer = stdoutBuffer.slice(newlineIdx + 1);
          processLine(line);
        }
      });

      child.stderr?.on("data", (chunk: Buffer) => {
        stderrBuffer += chunk.toString("utf8");
      });

      child.on("error", (err: Error) => {
        clearTimeout(timeoutHandle);
        reject(
          new Error(
            `Failed to spawn codex CLI (${config.binary}): ${err.message}`,
          ),
        );
      });

      child.on("close", (code: number | null) => {
        clearTimeout(timeoutHandle);

        // Drain any trailing partial line
        if (stdoutBuffer.trim().length > 0) {
          processLine(stdoutBuffer);
        }

        if (failureMessage) {
          reject(new Error(`Codex CLI error: ${failureMessage}`));
          return;
        }

        if (code !== 0 && agentMessageTexts.length === 0) {
          const stderrPreview = stderrBuffer.slice(0, 500);
          reject(
            new Error(`Codex CLI exited with code ${code}: ${stderrPreview}`),
          );
          return;
        }

        const rawContent = agentMessageTexts.join("\n\n");
        const content = this.stripMarkdownFences(rawContent);

        const totalTokens =
          (usage?.input_tokens || 0) + (usage?.output_tokens || 0);

        if (totalTokens > 0) {
          const inputTokens = usage?.input_tokens || 0;
          const outputTokens = usage?.output_tokens || 0;
          const cachedTokens = usage?.cached_input_tokens || 0;
          logger.info("SDK", "Codex CLI usage", {
            model: config.model,
            inputTokens,
            cachedInputTokens: cachedTokens,
            outputTokens,
            totalTokens,
            messagesInContext: truncatedHistory.length,
            threadId,
          });

          if (totalTokens > 50000) {
            logger.warn(
              "SDK",
              "High Codex token usage detected — consider reducing context",
              {
                totalTokens,
              },
            );
          }
        }

        resolve({ content, tokensUsed: totalTokens, threadId });
      });
    });
  }

  private getCodexConfig(): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);

    const authMethod = (settings.CLAUDE_MEM_CODEX_AUTH_METHOD || "cli") as
      | "cli"
      | "api";
    const apiKey =
      settings.CLAUDE_MEM_CODEX_API_KEY ||
      getCredential("OPENAI_API_KEY") ||
      "";
    const model = settings.CLAUDE_MEM_CODEX_MODEL || DEFAULT_CODEX_MODEL;
    const reasoning =
      settings.CLAUDE_MEM_CODEX_REASONING || DEFAULT_CODEX_REASONING;
    const binary = settings.CLAUDE_MEM_CODEX_BINARY || "codex";

    return { authMethod, apiKey, model, reasoning, binary };
  }

  private async ensureCodexAvailable(config: CodexConfig): Promise<void> {
    if (config.authMethod === "api" && !config.apiKey) {
      throw new Error(
        "Codex API auth selected but no OPENAI_API_KEY configured. Set CLAUDE_MEM_CODEX_API_KEY in settings or OPENAI_API_KEY environment variable.",
      );
    }
    // CLI mode relies on `codex login` having been run; the spawn itself
    // will surface auth errors if the stored token is invalid.
  }
}

interface CodexConfig {
  authMethod: "cli" | "api";
  apiKey: string;
  model: string;
  reasoning: string;
  binary: string;
}

/**
 * Check if Codex provider is available.
 * CLI mode: presence of `codex` binary is assumed until spawn fails.
 * API mode: requires OPENAI_API_KEY via settings or centralized ~/.claude-mem/.env.
 */
export function isCodexAvailable(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  const authMethod = settings.CLAUDE_MEM_CODEX_AUTH_METHOD || "cli";
  if (authMethod === "cli") {
    // We cannot probe the binary here without side effects; availability is
    // asserted on first spawn. Users on CLI auth without `codex login` done
    // will see an error and can switch provider back.
    return true;
  }
  return !!(
    settings.CLAUDE_MEM_CODEX_API_KEY || getCredential("OPENAI_API_KEY")
  );
}

/**
 * Check if Codex is the selected provider.
 */
export function isCodexSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === "codex";
}
