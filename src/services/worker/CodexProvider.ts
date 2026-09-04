import { spawn } from 'child_process';
import { existsSync, mkdirSync, readlinkSync, rmSync, symlinkSync } from 'fs';
import { homedir, tmpdir } from 'os';
import { delimiter, join } from 'path';
import { logger } from '../../utils/logger.js';
import { SettingsDefaultsManager, type SettingsDefaults } from '../../shared/SettingsDefaultsManager.js';
import { expandTilde, paths } from '../../shared/paths.js';
import { sanitizeEnv } from '../../supervisor/env-sanitizer.js';
import { waitForSlot } from '../../supervisor/process-registry.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { ClassifiedProviderError, type ProviderErrorClass } from './provider-errors.js';
import { withRetry } from './retry.js';
import { OpenAICompatibleProvider, type ProviderQueryResult } from './OpenAICompatibleProvider.js';

/**
 * Codex provider: memory generation on the user's ChatGPT plan via the local
 * `codex exec` CLI, which authenticates out-of-band through its own OAuth.
 *
 * `codex exec` is an agent, not a chat endpoint: it always ships its system
 * prompt plus its built-in skills, so a measured call costs 5-10 s and
 * ~16-19 k input tokens no matter how small the prompt is. That is the price
 * of the OAuth, not a misconfiguration — `--disable skills` does not exist and
 * plugins cannot be switched off from the command line.
 */

/** Sized for the tail of a 5-10 s agent turn, not for an HTTP round trip. */
export const CODEX_EXEC_TIMEOUT_MS = 120_000;

const CODEX_BINARY_NAME = 'codex';
const DEFAULT_CODEX_MODEL = 'gpt-5.4-mini';
const DEFAULT_CODEX_REASONING_EFFORT = 'low';

/** The isolated CODEX_HOME lives in the data dir, beside the other worker state. */
const CODEX_HOME_DIRNAME = 'codex-home';
const CODEX_AUTH_FILENAME = 'auth.json';
const USER_CODEX_AUTH_PATH = join(homedir(), '.codex', CODEX_AUTH_FILENAME);

const CODEX_PATH_REMEDIATION =
  'Install the Codex CLI and run `codex login`, or set CLAUDE_MEM_CODEX_PATH in ~/.claude-mem/settings.json.';

/**
 * Config for one `codex exec` invocation.
 *
 * `apiKey` is NOT a credential: it carries the resolved `codex` binary path
 * ('' when none was found). It is named that way to satisfy the base class's
 * `{ apiKey: string }` bound, whose `if (!apiKey) throw missingApiKeyError()`
 * gate is exactly the "codex CLI is missing" condition here. Never route it
 * through credential handling — the real secret stays in the user's
 * ~/.codex/auth.json and is never read by this process.
 */
export interface CodexConfig {
  apiKey: string;
  model: string;
  reasoningEffort: string;
}

/**
 * Resolve the `codex` binary: the explicit override first, then a PATH scan.
 *
 * Mirrors CLAUDE_CODE_PATH's semantics (override wins, leading `~` expanded
 * because the path is fed to existsSync/posix_spawn with no shell), but
 * deliberately does not reuse find-claude-executable: its version probing and
 * desktop-app detection are Claude-CLI specific.
 */
export function resolveCodexBinary(
  settings: SettingsDefaults = SettingsDefaultsManager.loadFromFile(paths.settings()),
): string {
  // Coerced, not trusted: loadFromFile assigns persisted values raw
  // (Record<string, any>) and POST /api/settings does not coerce them, so a
  // hand-edited `null` would throw a TypeError out of here — out of
  // selectLocalProvider, which is on the path of /api/health and every ingest.
  // Same guard as SettingsRoutes' CLAUDE_CODE_PATH.
  const configured = String(settings.CLAUDE_MEM_CODEX_PATH ?? '').trim();
  if (configured) {
    const expanded = expandTilde(configured);
    if (existsSync(expanded)) {
      return expanded;
    }
    logger.warn(
      'SDK',
      `CLAUDE_MEM_CODEX_PATH is set to "${configured}" but the file does not exist; falling back to a PATH scan`,
    );
  }

  for (const dir of (process.env.PATH ?? '').split(delimiter).filter(Boolean)) {
    const candidate = join(dir, CODEX_BINARY_NAME);
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return '';
}

const CODEX_FLATTEN_PREAMBLE =
  'Below is a memory-extraction conversation. Answer ONLY the final USER turn, ' +
  'in exactly the output format that turn asks for. The earlier turns are ' +
  'context. Emit nothing else: no preamble, no commentary, no tool calls.';

/**
 * Collapse the multi-turn history into one prompt. `codex exec` takes a single
 * prompt on stdin and has no conversation API, so the roles are carried as
 * headings and the preamble names the turn that must be answered.
 */
export function flattenHistory(history: ConversationMessage[]): string {
  const turns = history.map((message) => `### ${message.role.toUpperCase()}\n${message.content}`);
  return [CODEX_FLATTEN_PREAMBLE, ...turns].join('\n\n');
}

interface CodexUsage {
  input_tokens?: number;
  output_tokens?: number;
}

interface CodexEvent {
  type?: string;
  item?: { type?: string; text?: string; message?: string };
  usage?: CodexUsage;
  message?: string;
  error?: { message?: string };
}

function codexEventMessage(event: CodexEvent): string {
  return event.error?.message ?? event.message ?? 'codex exec reported a failed turn';
}

function parseCodexEvents(stdout: string): CodexEvent[] {
  const events: CodexEvent[] = [];
  for (const line of stdout.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      events.push(JSON.parse(trimmed) as CodexEvent);
    } catch {
      // [ANTI-PATTERN IGNORED]: codex interleaves human-readable progress lines
      // with the JSONL stream; a line that does not parse is not an event.
      continue;
    }
  }
  return events;
}

/**
 * The diagnostic for a failed run, read off STDOUT.
 *
 * Measured on codex-cli 0.153.1: every API-level failure writes its message to
 * stdout as `turn.failed` and leaves the real reason off stderr entirely — a
 * bad model and a bad reasoning_effort both exit 1 with ZERO bytes of stderr,
 * and an injected 429 exits 1 with the quota wording on stdout while stderr
 * carries only websocket tracing noise. Classifying from stderr therefore
 * cannot see quota or rate limits, so the breaker never arms and every
 * captured tool call buys the same refusal at ~16-19 k input tokens (#3634).
 *
 * `turn.failed` is preferred over the top-level `error` events because those
 * also carry codex's non-fatal reconnect ladder; the last one wins because a
 * run that emitted several ended on the last.
 */
export function codexFailureMessage(stdout: string): string | undefined {
  const events = parseCodexEvents(stdout);
  const failed = events.filter((event) => event.type === 'turn.failed');
  if (failed.length > 0) return codexEventMessage(failed[failed.length - 1]);

  const errors = events.filter((event) => event.type === 'error');
  if (errors.length > 0) return codexEventMessage(errors[errors.length - 1]);

  return undefined;
}

/**
 * The message a non-zero `codex exec` exit is classified from.
 *
 * Exported so the only copy of this fallback ORDER is the one production runs:
 * stdout carries the reason (see codexFailureMessage) and stderr is the
 * fallback for failures that never reach a turn — a spawn error, a config
 * parse error. Reversing the two files a quota-exhausted run as a rate limit,
 * which arms the wrong breaker.
 */
export function codexExitMessage(
  stdout: string,
  stderr: string,
  code: number | null,
  signal?: NodeJS.Signals | null,
): string {
  return (
    codexFailureMessage(stdout) ||
    stderr.trim() ||
    `codex exec exited with code ${code}${signal ? ` (signal ${signal})` : ''}`
  );
}

/**
 * Parse the `codex exec --json` event stream into a normalized result.
 *
 * The trap has two floors, and NEITHER kind of `error` event is fatal:
 *  - an `item.completed` whose `item.type` is 'error' is a warning (e.g.
 *    "skill descriptions were shortened", "falling back from WebSockets");
 *  - a TOP-LEVEL `{"type":"error"}` is how codex reports its 5-step reconnect
 *    ladder and its WebSocket-to-HTTPS transport fallback MID-TURN. Measured:
 *    five `Reconnecting... N/5` events on a turn that went on to emit its
 *    agent_message and exit 0. Recovering after a blip is the designed path,
 *    so treating it as fatal turns a success into a failure — and because
 *    forwardEmptyMessageResponse is false, leaves the batch queued.
 *
 * Only `turn.failed` is a real failure.
 */
export function parseCodexJsonl(stdout: string): ProviderQueryResult {
  let content = '';
  let usage: CodexUsage | undefined;

  for (const event of parseCodexEvents(stdout)) {
    if (event.type === 'item.completed') {
      if (event.item?.type === 'agent_message' && typeof event.item.text === 'string') {
        // Last one wins: a turn may emit several messages, the final is the answer.
        content = event.item.text;
      } else if (event.item?.type === 'error') {
        logger.debug('SDK', 'Codex reported a non-fatal error item', {
          message: event.item.message,
        });
      }
      continue;
    }

    if (event.type === 'turn.completed') {
      usage = event.usage;
      continue;
    }

    if (event.type === 'error') {
      logger.debug('SDK', 'Codex reported a non-fatal top-level error event', {
        message: event.message,
      });
      continue;
    }

    if (event.type === 'turn.failed') {
      throw classifyCodexError(codexEventMessage(event));
    }
  }

  return {
    content,
    tokensUsed: usage ? (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0) : undefined,
    inputTokens: usage?.input_tokens,
    outputTokens: usage?.output_tokens,
  };
}

// Narrowed to what "setup_required" actually means. It used to carry bare
// `auth` and bare `login`, and every codex transport failure says "401
// Unauthorized" — so a plan limit arriving alongside an authorization line was
// swallowed into the setup branch, whose re-probe (isCodexAvailable) only
// resolves the binary and always says yes, clearing after 30 s and retrying
// against the same exhausted quota forever.
const CODEX_SETUP_REQUIRED_PATTERN = /ENOENT|EACCES|^spawn |not logged in|codex login|401 Unauthorized/i;

/**
 * Every allowance-exhausted wording the shipped binary can print, read off
 * `strings -a` on codex-cli 0.153.1 rather than guessed. A personal plan says
 * "You've hit your usage limit."; a workspace one says "You're out of
 * credits.", "You've reached your workspace credit limit" or "You hit your
 * spend cap set in your workspace."; the app-server reports the same states as
 * the `workspace_{owner,member}_credits_depleted` tags. All of them mean the
 * same thing to us — arm the breaker — and a miss re-opens the hole where the
 * failure is filed 'unrecoverable', recordQuotaExhausted never fires, and the
 * next batch walks straight back into the exhausted allowance.
 */
const CODEX_QUOTA_PATTERN = /usage limit|usage_limit|quota|out of credits|credit limit|spend cap|credits_depleted/i;
const CODEX_RATE_LIMIT_PATTERN = /rate.?limit|429|too many requests/i;

/**
 * Transient covers the whole network-death vocabulary, not just timeouts:
 * measured with the endpoint unreachable, codex's last top-level error is
 * "Reconnecting... waiting for network (Connection failed: error sending
 * request)", and its stream drops read "stream disconnected before
 * completion". Filed 'unrecoverable' those were never retried, dropping the
 * observation batch; withRetry is maxRetries 1, so the cost of being wrong
 * here is capped at one extra attempt.
 *
 * The 5xx alternative is keyed off the STRUCTURED form codex prints ("status
 * 503", "HTTP error: 503") instead of scanning prose for three digits: a bare
 * `5\d\d` matched the "591" inside the tracing timestamp
 * "2026-09-04T16:39:08.182591Z" and read a terminal failure as retryable.
 */
const CODEX_TRANSIENT_PATTERN =
  /timeout|ECONNRESET|ECONNREFUSED|(?:^|\W)(?:status|HTTP error)\D{0,12}5\d\d\b|temporar|connection (?:refused|failed)|error sending request|stream disconnected/i;

/**
 * Opaque ids codex appends to transport prose. Measured message:
 * "...cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b"
 * — that id contains the run "581", read as a 5xx status back when
 * CODEX_TRANSIENT_PATTERN scanned prose for a bare `5\d\d`, so a terminal
 * failure was retried for another ~25 k tokens before failing anyway.
 *
 * Kept as belt and braces now that the 5xx alternative keys off the structured
 * "status 5xx" form: the scrub is free and still keeps every other pattern
 * from reading meaning out of a random hex id.
 */
const CODEX_OPAQUE_ID_PATTERN = /\b(?:request id|cf-ray):\s*\S+/gi;

function codexErrorKind(message: string): ProviderErrorClass {
  const scrubbed = message.replace(CODEX_OPAQUE_ID_PATTERN, '');
  // Quota and rate limit are tested FIRST: they are the classifications with a
  // breaker behind them, and their wording routinely arrives next to an
  // authorization line that the setup pattern would otherwise claim.
  if (CODEX_QUOTA_PATTERN.test(scrubbed)) return 'quota_exhausted';
  if (CODEX_RATE_LIMIT_PATTERN.test(scrubbed)) return 'rate_limit';
  if (CODEX_SETUP_REQUIRED_PATTERN.test(scrubbed)) return 'setup_required';
  if (CODEX_TRANSIENT_PATTERN.test(scrubbed)) return 'transient';
  return 'unrecoverable';
}

/**
 * Classify a `codex exec` failure from its message. There is no status code to
 * read: the CLI reports a spawn failure, an auth problem and a plan limit as
 * prose, mostly inside a `turn.failed` event on stdout.
 *
 * The plan-limit wording is no longer a guess — codex emits
 * "You've hit your usage limit. Try again later.", which CODEX_QUOTA_PATTERN
 * matches on "usage limit".
 */
export function classifyCodexError(message: string, cause?: unknown): ClassifiedProviderError {
  const kind = codexErrorKind(message);
  if (kind === 'unrecoverable') {
    // Carry the raw message at WARN so any wording the patterns above do not
    // yet know can be read off a log and folded in, instead of being guessed.
    logger.warn('SDK', 'Unclassified codex failure; recording the raw message so the wording can be learned', {
      message,
    });
  }
  return new ClassifiedProviderError(message, {
    kind,
    cause: cause ?? new Error(message),
  });
}

export function isCodexAvailable(): boolean {
  return resolveCodexBinary() !== '';
}

export function isCodexSelected(): boolean {
  return SettingsDefaultsManager.loadFromFile(paths.settings()).CLAUDE_MEM_PROVIDER === 'codex';
}

function readAuthLinkTarget(linkPath: string): string | null {
  try {
    return readlinkSync(linkPath);
  } catch {
    // [ANTI-PATTERN IGNORED]: readlink fails with ENOENT (no link yet) and
    // EINVAL (a regular file left by an earlier version); both mean "not the
    // link we want" and are repaired by the caller.
    return null;
  }
}

/**
 * Create, idempotently, an isolated CODEX_HOME holding nothing but a symlink
 * to the user's real auth.json, and return its path.
 *
 * A memory worker must inherit none of the user's own codex setup: their
 * config.toml can set approval_policy=never, sandbox danger-full-access and a
 * `notify` program that launches a GUI app, their hooks.json would fire on
 * every single generation, and generating inside ~/.codex would put memory
 * generation into the very memories directory it is summarizing.
 *
 * SYMLINK, not copy: codex writes the refreshed OAuth token back to auth.json,
 * so a copy goes stale and auth silently breaks days later.
 *
 * COST, measured on codex-cli 0.153.1: a fresh home is NOT the couple of MB an
 * auth symlink suggests. Two generations grew it to 33 MB, 27 MB of it
 * `plugins/` — a home with no plugins makes codex RE-INSTALL them, and a
 * `.remote-plugin-install-staging/` appears, i.e. the memory worker fetches
 * remote plugin code (openai-curated-remote: github, slack, gmail, notion,
 * google-calendar, ...) over the network on its first generations. How far it
 * gets is a race against process exit — one measured home stopped at 76 KB
 * because `codex exec` returned first — so the ceiling, not the average, is
 * what to budget for.
 *
 * Suppressing it was probed and did not work: pre-seeding an empty `plugins/`
 * directory does NOT stop the fetch (that home still reached 26 MB, on its
 * first generation), and `-c plugins.enabled=false` is rejected by the config
 * parser ("invalid type: boolean, expected struct PluginConfig"). So this is
 * the price of the isolation, not a bug to fix here.
 */
export function ensureIsolatedCodexHome(): string {
  if (!existsSync(USER_CODEX_AUTH_PATH)) {
    throw new ClassifiedProviderError(
      `Codex is not logged in: ${USER_CODEX_AUTH_PATH} does not exist. Run \`codex login\` and retry.`,
      { kind: 'setup_required', cause: new Error('codex auth.json missing') },
    );
  }

  const codexHome = join(paths.dataDir(), CODEX_HOME_DIRNAME);
  const linkPath = join(codexHome, CODEX_AUTH_FILENAME);
  mkdirSync(codexHome, { recursive: true });

  if (readAuthLinkTarget(linkPath) !== USER_CODEX_AUTH_PATH) {
    rmSync(linkPath, { force: true });
    symlinkSync(USER_CODEX_AUTH_PATH, linkPath);
  }
  return codexHome;
}

/**
 * `--ignore-user-config` is kept on top of the isolated CODEX_HOME: it costs
 * nothing and still protects if CODEX_HOME is ever pointed at the real dir.
 * The prompt goes on stdin (the trailing `-`), never in argv, so no length or
 * quoting limit applies to it.
 */
function buildCodexArgv(config: CodexConfig): string[] {
  return [
    'exec',
    '--json',
    '--ephemeral',
    '--ignore-user-config',
    '-s',
    'read-only',
    '--skip-git-repo-check',
    '-C',
    tmpdir(),
    '-m',
    config.model,
    '-c',
    `model_reasoning_effort="${config.reasoningEffort}"`,
    '-',
  ];
}

export class CodexProvider extends OpenAICompatibleProvider<CodexConfig> {
  // Must stay literally 'Codex': ResponseProcessor maps this name onto the
  // telemetry provider enum.
  protected readonly providerName = 'Codex';
  protected readonly syntheticIdPrefix = 'codex';
  // Mirrors Gemini: empty output leaves the batch queued rather than
  // forwarding an empty response to the parser.
  protected readonly forwardEmptyMessageResponse = false;

  protected getConfig(): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(paths.settings());
    return {
      apiKey: resolveCodexBinary(settings),
      model: settings.CLAUDE_MEM_CODEX_MODEL || DEFAULT_CODEX_MODEL,
      reasoningEffort: settings.CLAUDE_MEM_CODEX_REASONING_EFFORT || DEFAULT_CODEX_REASONING_EFFORT,
    };
  }

  /**
   * Classified, not a bare Error: SessionRoutes' setup_required branch is what
   * stops a fresh generator being spawned per ingest while the CLI is missing.
   */
  protected missingApiKeyError(): Error {
    return new ClassifiedProviderError(`codex CLI not found. ${CODEX_PATH_REMEDIATION}`, {
      kind: 'setup_required',
      cause: new Error('codex binary not resolved'),
    });
  }

  /**
   * Field compression is disabled on codex, so no child is ever spawned for it.
   *
   * Measured: one real field-compression call (29 KB payload, budget 12800)
   * took 59,493 ms and 24,849 input + 3,883 output tokens — against
   * FIELD_OPTIMIZE_TIMEOUT_MS of 30_000, which field-optimizer enforces with a
   * bare Promise.race that abandons the attempt WITHOUT killing the child. The
   * orphan then holds its `waitForSlot` reservation for up to
   * CODEX_EXEC_TIMEOUT_MS — in the pool ClaudeProvider shares, so at the
   * default CLAUDE_MEM_MAX_CONCURRENT_AGENTS=2 one observation with two
   * oversized fields reserves both slots and blocks its own observation query
   * behind its own orphans. The attempt could therefore only ever time out.
   *
   * Returning null lands on field-optimizer's existing truncation fallback:
   * the same end state, minus ~29 k wasted tokens per attempt and the stall.
   * Revisit if either number moves — the ceiling is 59.5 s against a 30 s
   * budget.
   */
  protected compressField(): Promise<string | null> {
    return Promise.resolve(null);
  }

  /** Contract only — the base class never calls this for a codex session. */
  protected estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  protected buildLastUsage(result: ProviderQueryResult): ActiveSession['lastUsage'] {
    // Both sides or nothing: a backend reporting only one of the two counts
    // must not produce a half-real event (input=0 → compression_ratio 0.0).
    return typeof result.inputTokens === 'number' && typeof result.outputTokens === 'number'
      ? { input: result.inputTokens, output: result.outputTokens }
      : null;
  }

  protected async query(history: ConversationMessage[], config: CodexConfig): Promise<ProviderQueryResult> {
    const prompt = flattenHistory(history);
    // No session AbortSignal reaches query() — the same ceiling as Gemini and
    // OpenRouter, whose withRetry calls also omit abortSignal. For a
    // subprocess that means a session abort does not kill an in-flight child;
    // only the per-attempt timeout below ends it.
    return withRetry((attemptSignal) => this.runCodexExec(prompt, config, attemptSignal), {
      label: `Codex ${config.model}`,
      maxRetries: 1,
      perAttemptTimeoutMs: CODEX_EXEC_TIMEOUT_MS,
    });
  }

  private async runCodexExec(prompt: string, config: CodexConfig, signal: AbortSignal): Promise<ProviderQueryResult> {
    const codexHome = ensureIsolatedCodexHome();
    const settings = SettingsDefaultsManager.loadFromFile(paths.settings());
    const maxConcurrent = parseInt(settings.CLAUDE_MEM_MAX_CONCURRENT_AGENTS, 10) || 2;
    // Codex children share the Claude agents' process pool: unbounded, dozens
    // of live sessions would each spawn their own codex exec (#3287 is the
    // same failure on the SDK path). The reservation is held for the child's
    // whole life because it is never registered as an SDK process, so nothing
    // else accounts for it.
    const slot = await waitForSlot(maxConcurrent, signal);

    try {
      return await new Promise<ProviderQueryResult>((resolve, reject) => {
        const child = spawn(config.apiKey, buildCodexArgv(config), {
          cwd: tmpdir(),
          env: { ...sanitizeEnv(process.env), CODEX_HOME: codexHome },
          stdio: ['pipe', 'pipe', 'pipe'],
          windowsHide: true,
        });

        let stdout = '';
        let stderr = '';
        child.stdout.setEncoding('utf-8');
        child.stderr.setEncoding('utf-8');
        child.stdout.on('data', (chunk: string) => {
          stdout += chunk;
        });
        child.stderr.on('data', (chunk: string) => {
          stderr += chunk;
        });

        const onAbort = (): void => {
          child.kill('SIGTERM');
        };
        signal.addEventListener('abort', onAbort, { once: true });

        child.on('error', (error: Error) => {
          signal.removeEventListener('abort', onAbort);
          reject(classifyCodexError(error.message, error));
        });

        child.on('close', (code: number | null, closeSignal: NodeJS.Signals | null) => {
          signal.removeEventListener('abort', onAbort);

          if (signal.aborted) {
            const aborted = new Error(`Codex exec aborted after ${CODEX_EXEC_TIMEOUT_MS}ms`);
            aborted.name = 'AbortError';
            reject(aborted);
            return;
          }

          if (code !== 0) {
            reject(classifyCodexError(codexExitMessage(stdout, stderr, code, closeSignal)));
            return;
          }

          try {
            resolve(parseCodexJsonl(stdout));
          } catch (parseError: unknown) {
            reject(parseError);
          }
        });

        // A destroyed pipe (a failed spawn, or the SIGTERM above landing while
        // a long prompt is still flushing) emits EPIPE on stdin. Unhandled,
        // that crashes the worker; the real cause is reported by the
        // 'error'/'close' handlers above.
        child.stdin.on('error', (error: Error) => {
          logger.debug('SDK', 'Codex stdin closed before the prompt was written', { error: error.message });
        });
        child.stdin.end(prompt);
      });
    } finally {
      slot.release();
    }
  }
}
