import { afterEach, beforeEach, describe, it, expect, mock, spyOn } from 'bun:test';
import {
  classifyCodexError,
  codexExitMessage,
  codexFailureMessage,
  CodexProvider,
  flattenHistory,
  parseCodexJsonl,
  type CodexConfig,
} from '../../src/services/worker/CodexProvider.js';
import { isClassified } from '../../src/services/worker/provider-errors.js';
import type { ProviderQueryResult } from '../../src/services/worker/OpenAICompatibleProvider.js';
import { ModeManager } from '../../src/services/domain/ModeManager.js';
import { SettingsDefaultsManager } from '../../src/shared/SettingsDefaultsManager.js';
import type { DatabaseManager } from '../../src/services/worker/DatabaseManager.js';
import { SessionManager } from '../../src/services/worker/SessionManager.js';
import type { ActiveSession, ConversationMessage } from '../../src/services/worker-types.js';

// The success fixture is the real event stream of a `codex exec --json` run,
// including the NON-FATAL warning codex emits as an `item.completed` whose
// `item.type` is 'error'. That turn succeeded: treating the inner error item as
// fatal would fail every generation.
const CODEX_SUCCESS_JSONL = [
  '{"type":"thread.started","thread_id":"01a06cfb-269d-7373-93cb-864322f750fb"}',
  '{"type":"turn.started"}',
  '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Skill descriptions were shortened to fit the skills context budget."}}',
  '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"first draft"}}',
  '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"<observation><type>discovery</type></observation>"}}',
  '{"type":"turn.completed","usage":{"input_tokens":19554,"cached_input_tokens":4352,"cache_write_input_tokens":0,"output_tokens":42,"reasoning_output_tokens":16}}',
  '',
].join('\n');

/**
 * Verbatim stdout of a real `codex exec --json` run (codex-cli 0.153.1) whose
 * websocket transport was rejected, so codex walked its 5-step reconnect
 * ladder, fell back to HTTPS, hit one more stream drop — and then finished the
 * turn and EXITED 0. Every `Reconnecting...` line here is a TOP-LEVEL
 * `{"type":"error"}` event: the second, non-obvious floor of the non-fatal-error
 * trap. Treating them as fatal turns this success into a failure.
 */
const CODEX_RECONNECT_RECOVERY_JSONL = [
  '{"type":"thread.started","thread_id":"01a06d36-a77f-7821-89d2-980aac29de3e"}',
  '{"type":"turn.started"}',
  '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Skill descriptions were shortened to fit the skills context budget. Codex can still see every skill, but some descriptions are shorter. Disable unused skills or plugins to leave more room for the rest."}}',
  '{"type":"error","message":"Reconnecting... 2/5 (unexpected status 401 Unauthorized: Unknown error, url: ws://127.0.0.1:8080/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b)"}',
  '{"type":"error","message":"Reconnecting... 3/5 (unexpected status 401 Unauthorized: Unknown error, url: ws://127.0.0.1:8080/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b)"}',
  '{"type":"error","message":"Reconnecting... 4/5 (unexpected status 401 Unauthorized: Unknown error, url: ws://127.0.0.1:8080/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b)"}',
  '{"type":"error","message":"Reconnecting... 5/5 (unexpected status 401 Unauthorized: Unknown error, url: ws://127.0.0.1:8080/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b)"}',
  '{"type":"item.completed","item":{"id":"item_1","type":"error","message":"Falling back from WebSockets to HTTPS transport. unexpected status 401 Unauthorized: Unknown error, url: ws://127.0.0.1:8080/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b"}}',
  '{"type":"error","message":"Reconnecting... 1/5 (stream disconnected before completion: Transport error: network error: error decoding response body)"}',
  '{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"OK"}}',
  '{"type":"turn.completed","usage":{"input_tokens":1234,"cached_input_tokens":0,"cache_write_input_tokens":0,"output_tokens":7,"reasoning_output_tokens":0}}',
  '',
].join('\n');

/**
 * Verbatim stdout of a real run that hit a plan limit: exit 1, the reason on
 * STDOUT as `turn.failed`, and stderr carrying only websocket tracing noise
 * (measured 599 bytes, none of it the quota message). This is the exact stream
 * the quota breaker has to classify from.
 */
const CODEX_QUOTA_FAILURE_JSONL = [
  '{"type":"thread.started","thread_id":"01a06d3a-e7bd-77e0-8954-d27d4d612e2b"}',
  '{"type":"turn.started"}',
  '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Skill descriptions were shortened to fit the skills context budget."}}',
  '{"type":"error","message":"You\'ve hit your usage limit. Try again later."}',
  '{"type":"turn.failed","error":{"message":"You\'ve hit your usage limit. Try again later."}}',
  '',
].join('\n');

/** Real stderr of that same run: 429 tracing lines, no quota wording. */
const CODEX_QUOTA_FAILURE_STDERR = [
  '2026-09-04T16:22:52.088985Z ERROR codex_models_manager::manager: failed to refresh available models: stream disconnected before completion',
  '2026-09-04T16:22:52.591474Z ERROR codex_api::endpoint::responses_websocket: failed to connect to websocket: HTTP error: 429 Too Many Requests, url: ws://127.0.0.1:8081/v1/responses',
  '',
].join('\n');

/**
 * Verbatim stdout of `codex exec ... -m definitely-not-a-model`: exit 1,
 * turn.failed on stdout, and ZERO bytes of stderr.
 */
const CODEX_BAD_MODEL_JSONL = [
  '{"type":"thread.started","thread_id":"01a06d2d-8275-7670-9636-1f9e887e2a2a"}',
  '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Model metadata for `definitely-not-a-model` not found. Defaulting to fallback metadata; this can degrade performance and cause issues."}}',
  '{"type":"turn.started"}',
  '{"type":"item.completed","item":{"id":"item_1","type":"error","message":"Skill descriptions were shortened to fit the skills context budget. Codex can still see every skill, but some descriptions are shorter. Disable unused skills or plugins to leave more room for the rest."}}',
  '{"type":"error","message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":{\\"type\\":\\"invalid_request_error\\",\\"message\\":\\"The \'definitely-not-a-model\' model is not supported when using Codex with a ChatGPT account.\\"}}"}',
  '{"type":"turn.failed","error":{"message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":{\\"type\\":\\"invalid_request_error\\",\\"message\\":\\"The \'definitely-not-a-model\' model is not supported when using Codex with a ChatGPT account.\\"}}"}}',
  '',
].join('\n');

/**
 * The exact pair `runCodexExec`'s close handler runs on a non-zero exit. It
 * CALLS codexExitMessage rather than restating its `||` chain: a copy here
 * would keep passing while production was reordered to stderr-first, which is
 * the one regression this file exists to catch.
 */
function nonZeroExitKind(stdout: string, stderr: string, code = 1): string {
  return classifyCodexError(codexExitMessage(stdout, stderr, code)).kind;
}

describe('parseCodexJsonl', () => {
  it('keeps the last agent_message and the turn usage, ignoring a non-fatal error item', () => {
    const result = parseCodexJsonl(CODEX_SUCCESS_JSONL);

    expect(result.content).toBe('<observation><type>discovery</type></observation>');
    expect(result.inputTokens).toBe(19554);
    expect(result.outputTokens).toBe(42);
    expect(result.tokensUsed).toBe(19596);
  });

  it('survives codex\'s top-level reconnect ladder on a turn that then succeeds', () => {
    const result = parseCodexJsonl(CODEX_RECONNECT_RECOVERY_JSONL);

    expect(result.content).toBe('OK');
    expect(result.inputTokens).toBe(1234);
    expect(result.outputTokens).toBe(7);
    expect(result.tokensUsed).toBe(1241);
  });

  // Belt and braces only: production never reaches the parser on a failed run,
  // because a turn.failed always comes with a non-zero exit that
  // `runCodexExec` rejects first (see the non-zero-exit tests below).
  it('throws a classified error if a turn.failed somehow arrives on a clean exit', () => {
    const jsonl = '{"type":"turn.failed","error":{"message":"model stream disconnected"}}';

    expect(() => parseCodexJsonl(jsonl)).toThrow('model stream disconnected');
  });
});

describe('codexFailureMessage', () => {
  it('reads the turn.failed diagnostic off stdout', () => {
    expect(codexFailureMessage(CODEX_QUOTA_FAILURE_JSONL))
      .toBe("You've hit your usage limit. Try again later.");
  });

  it('falls back to the last top-level error when no turn was reached', () => {
    const jsonl = '{"type":"error","message":"Reconnecting... 1/5 (a)"}\n{"type":"error","message":"Reconnecting... 2/5 (b)"}';

    expect(codexFailureMessage(jsonl)).toBe('Reconnecting... 2/5 (b)');
  });

  it('returns undefined for a stream that reported no failure', () => {
    expect(codexFailureMessage(CODEX_SUCCESS_JSONL)).toBeUndefined();
  });
});

describe('the non-zero-exit classification path', () => {
  it('arms the quota breaker from the stdout turn.failed, not from stderr', () => {
    expect(nonZeroExitKind(CODEX_QUOTA_FAILURE_JSONL, CODEX_QUOTA_FAILURE_STDERR))
      .toBe('quota_exhausted');
    // Preferring stderr would classify the same run as a rate limit — the
    // wrong breaker, because stderr only ever saw the websocket's 429.
    expect(classifyCodexError(CODEX_QUOTA_FAILURE_STDERR.trim()).kind).toBe('rate_limit');
  });

  it('classifies a stdout-only failure whose stderr is empty', () => {
    expect(nonZeroExitKind(CODEX_BAD_MODEL_JSONL, '')).toBe('unrecoverable');
    // The old behaviour: the literal exit-code string, classifiable as nothing.
    expect(classifyCodexError('codex exec exited with code 1').kind).toBe('unrecoverable');
  });

  it('still falls back to the exit code when codex produced no events at all', () => {
    expect(nonZeroExitKind('', '')).toBe('unrecoverable');
  });
});

describe('classifyCodexError', () => {
  it('maps codex\'s own usage-limit wording to quota_exhausted', () => {
    const error = classifyCodexError("You've hit your usage limit. Try again later.");

    expect(isClassified(error)).toBe(true);
    expect(error.kind).toBe('quota_exhausted');
  });

  // Every codex transport failure says "401 Unauthorized", so the setup branch
  // must not be allowed to claim a message that also carries a plan limit: its
  // re-probe is isCodexAvailable(), which only resolves the binary and always
  // says yes, so it would clear after 30 s and retry the exhausted quota.
  it('prefers quota and rate limit over setup_required when both appear', () => {
    expect(classifyCodexError("You've hit your usage limit. (401 Unauthorized)").kind)
      .toBe('quota_exhausted');
    expect(classifyCodexError('429 Too Many Requests (401 Unauthorized)').kind)
      .toBe('rate_limit');
  });

  // Every wording below is verbatim from `strings -a` on the shipped codex
  // binary (codex-cli 0.153.1). A workspace/Enterprise account never says
  // "usage limit": it exhausts credits or a spend cap, and each of these
  // arrives on stdout as turn.failed with exit 1. Missed, the failure is filed
  // 'unrecoverable', recordQuotaExhausted never fires, and the next batch
  // walks straight back into the exhausted allowance.
  it('maps the workspace credit and spend-cap wordings to quota_exhausted', () => {
    for (const message of [
      "You're out of credits.",
      'Your workspace is out of credits. Add credits to continue using Codex.',
      "You've reached your workspace credit limit",
      'You hit your spend cap set in your workspace. Increase your spend cap to continue.',
      'workspace_owner_credits_depleted',
    ]) {
      expect(classifyCodexError(message).kind).toBe('quota_exhausted');
    }
  });

  it('maps a missing binary to setup_required', () => {
    expect(classifyCodexError('spawn codex ENOENT').kind).toBe('setup_required');
    expect(classifyCodexError('Error: EACCES /usr/local/bin/codex').kind).toBe('setup_required');
    expect(classifyCodexError('You are not logged in. Run `codex login`.').kind).toBe('setup_required');
  });

  it('still treats a bare transport 401 as setup_required', () => {
    expect(classifyCodexError('unexpected status 401 Unauthorized, url: wss://api.openai.com/v1/responses').kind)
      .toBe('setup_required');
  });

  // Measured: the opaque request id below contains the run "581", which the
  // transient pattern's `5\d\d` read as a 5xx status — so a terminal failure
  // was retried, buying another 30-60 s and ~25 k tokens before failing anyway.
  it('does not read a 5xx status out of a request id or a cf-ray', () => {
    const terminal = 'Reconnecting... 2/5 (unexpected status 404 Not Found: Unknown error, url: ws://127.0.0.1:8083/v1/responses, cf-ray: a35e33760a04aaf7-YYZ, request id: req_6861cf726b144fc0afb7e12fd765819b)';

    expect(/5\d\d/.test(terminal)).toBe(true);
    expect(classifyCodexError(terminal).kind).not.toBe('transient');
  });

  // Measured on a run with the endpoint unreachable: codex's last top-level
  // error is a network death, and withRetry only retries transient|rate_limit.
  // Filed 'unrecoverable' these dropped the whole observation batch instead of
  // buying the one extra attempt maxRetries: 1 allows.
  it('classifies a connection death as transient so withRetry gets its one attempt', () => {
    for (const message of [
      'Reconnecting... waiting for network (Connection failed: error sending request)',
      'stream connection failed; waiting to retry',
      'stream disconnected before completion: Transport error: network error',
      'failed to connect: ECONNREFUSED 127.0.0.1:8080',
    ]) {
      expect(classifyCodexError(message).kind).toBe('transient');
    }
  });

  // Assembled from two measured stderr lines: the verbatim tracing prefix
  // whose fraction ".182591Z" carries "591", and a terminal 404. A bare
  // `5\d\d` read that timestamp as a 5xx status, so the 5xx alternative is now
  // keyed off the structured form codex actually prints.
  it('does not read a 5xx status out of a tracing timestamp', () => {
    const terminal =
      '2026-09-04T16:39:08.182591Z ERROR codex_models_manager::manager: failed to refresh available models: unexpected status 404 Not Found';

    expect(/5\d\d/.test(terminal)).toBe(true);
    expect(classifyCodexError(terminal).kind).not.toBe('transient');
  });

  it('still classifies a genuine 5xx as transient', () => {
    expect(classifyCodexError('unexpected status 503: upstream unavailable, request id: req_aaaa').kind)
      .toBe('transient');
    // The websocket formatter's own shape, measured with a 429 on a real run.
    expect(
      classifyCodexError('failed to connect to websocket: HTTP error: 503 Service Unavailable, url: ws://127.0.0.1:8081/v1/responses').kind,
    ).toBe('transient');
  });
});

describe('flattenHistory', () => {
  it('carries the preamble first and the final user turn last', () => {
    const flattened = flattenHistory([
      { role: 'user', content: 'init prompt' },
      { role: 'assistant', content: 'prior observation' },
      { role: 'user', content: 'answer this one' },
    ]);

    expect(flattened.startsWith('Below is a memory-extraction conversation.')).toBe(true);
    expect(flattened).toContain('### ASSISTANT\nprior observation');
    expect(flattened.endsWith('### USER\nanswer this one')).toBe(true);
  });
});

/**
 * Verbatim stdout of a real init-brief run (codex-cli 0.153.1, gpt-5.4-mini):
 * an `agent_message` whose text is EMPTY, after 124 reasoning tokens. That is
 * codex answering "nothing to record" — the shape the brief provokes, because
 * the brief carries the skip guidance ("return an empty response only") and no
 * tool call to record. Measured on 10 of 12 session starts.
 */
const CODEX_EMPTY_ANSWER_JSONL = [
  '{"type":"thread.started","thread_id":"01a06d83-aae3-7cd3-8601-ce37807dc4c5"}',
  '{"type":"turn.started"}',
  '{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Skill descriptions were shortened to fit the skills context budget."}}',
  '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":""}}',
  '{"type":"turn.completed","usage":{"input_tokens":18341,"cached_input_tokens":15616,"cache_write_input_tokens":0,"output_tokens":130,"reasoning_output_tokens":124}}',
  '',
].join('\n');

/** The same turn with no `agent_message` at all: codex said nothing, not "nothing". */
const CODEX_NO_ANSWER_JSONL = [
  '{"type":"thread.started","thread_id":"01a06d83-aae3-7cd3-8601-ce37807dc4c5"}',
  '{"type":"turn.started"}',
  '{"type":"turn.completed","usage":{"input_tokens":18341,"output_tokens":0}}',
  '',
].join('\n');

const mockMode = {
  name: 'code',
  prompts: { init: 'init', observation: 'obs', summary: 'summary' },
  observation_types: [{ id: 'discovery' }],
  observation_concepts: [],
};

/**
 * CodexProvider with only the codex child replaced: getConfig skips the binary
 * lookup and query records the flattened-history input it was handed. The
 * session lifecycle under test (init handling, message loop, empty-response
 * routing) is the real one.
 */
class StubbedCodexProvider extends CodexProvider {
  readonly sentHistories: ConversationMessage[][] = [];
  readonly sentModels: string[] = [];

  constructor(
    sessionManager: SessionManager,
    private readonly replies: string[],
  ) {
    super({} as unknown as DatabaseManager, sessionManager);
  }

  protected getConfig(): CodexConfig {
    return { apiKey: '/usr/bin/true', model: 'gpt-5.4-mini', reasoningEffort: 'low' };
  }

  protected async query(history: ConversationMessage[], config: CodexConfig): Promise<ProviderQueryResult> {
    this.sentHistories.push(history.map((message) => ({ ...message })));
    this.sentModels.push(config.model);
    return { content: this.replies.shift() ?? '' };
  }
}

function makeSession(): ActiveSession {
  return {
    sessionDbId: 1,
    contentSessionId: 'codex-session',
    memorySessionId: 'codex-session-123',
    project: 'ws',
    platformSource: 'claude',
    userPrompt: 'Ajoute un provider codex a claude-mem',
    abortController: new AbortController(),
    generatorPromise: null,
    lastPromptNumber: 1,
    startTime: Date.now(),
    cumulativeInputTokens: 0,
    cumulativeOutputTokens: 0,
    earliestPendingTimestamp: null,
    claimedMessageIds: [],
    conversationHistory: [],
    currentProvider: null,
    consecutiveRestarts: 0,
    consecutiveInvalidOutputs: 0,
    lastGeneratorActivity: Date.now(),
  };
}

/**
 * The mechanism guard for the two codex empty-response defects. Both are
 * asserted through the real startSession path, so they go red if the flags are
 * flipped back OR if the base class stops honoring them.
 *
 * 1. The init brief must not be sent as its own query. It asks for nothing
 *    actionable, so codex answers it with an empty message (10 of 12 measured
 *    session starts, one ERROR line each) or — twice in 12 — with an
 *    observation fabricated from `<user_request>` alone, which was then stored
 *    as if the work had happened.
 * 2. An empty observation reply must drain its batch. It is a skip verdict, and
 *    the in-RAM buffer has no durable queue behind it: a batch left claimed is
 *    destroyed by the idle teardown, storing nothing and logging nothing but a
 *    single WARN.
 */
describe('CodexProvider empty-response handling', () => {
  let modeManagerSpy: ReturnType<typeof spyOn>;
  let settingsSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    modeManagerSpy = spyOn(ModeManager, 'getInstance').mockImplementation(() => ({
      getActiveMode: () => mockMode,
      loadMode: () => {},
    } as unknown as ModeManager));
    settingsSpy = spyOn(SettingsDefaultsManager, 'loadFromFile').mockImplementation(
      () => SettingsDefaultsManager.getAllDefaults(),
    );
  });

  afterEach(() => {
    modeManagerSpy.mockRestore();
    settingsSpy.mockRestore();
    mock.restore();
  });

  function sessionManagerDouble(): { manager: SessionManager; confirmed: ReturnType<typeof mock> } {
    const confirmed = mock(() => Promise.resolve(1));
    const manager = {
      getMessageIterator: async function* () {
        yield { type: 'observation', tool_name: 'Bash', tool_input: { command: 'ls' }, tool_response: { stdout: '' }, prompt_number: 1 };
      },
      confirmClaimedMessages: confirmed,
      resetProcessingToPending: mock(() => Promise.resolve(0)),
    } as unknown as SessionManager;
    return { manager, confirmed };
  }

  it('sends one query per observation and never a standalone init query', async () => {
    const { manager } = sessionManagerDouble();
    const provider = new StubbedCodexProvider(manager, ['']);

    await provider.startSession(makeSession());

    expect(provider.sentHistories.length).toBe(1);
    expect(provider.sentHistories[0].map((message) => message.role)).toEqual(['user', 'user']);
    expect(provider.sentHistories[0][0].content).toContain('<user_request>');
    expect(provider.sentHistories[0][1].content).toContain('<observed_from_primary_session>');
  });

  it('drains the claimed batch when codex answers an observation with an empty message', async () => {
    const { manager, confirmed } = sessionManagerDouble();
    const provider = new StubbedCodexProvider(manager, ['']);

    await provider.startSession(makeSession());

    expect(confirmed).toHaveBeenCalledTimes(1);
  });

  it('reads an empty agent_message as empty content, not as a failure', () => {
    expect(parseCodexJsonl(CODEX_EMPTY_ANSWER_JSONL).content).toBe('');
  });

  it('never lets the Claude summary tier override the codex model', async () => {
    // Regression: CLAUDE_MEM_TIER_SUMMARY_MODEL names a Claude model, and codex
    // rejects it outright — "The 'claude-sonnet-4-6' model is not supported when
    // using Codex with a ChatGPT account." (400). Every summary failed while
    // observations, which never consult the tier, succeeded.
    settingsSpy.mockImplementation(() => ({
      ...SettingsDefaultsManager.getAllDefaults(),
      CLAUDE_MEM_TIER_ROUTING_ENABLED: 'true',
      CLAUDE_MEM_TIER_SUMMARY_MODEL: 'claude-sonnet-4-6',
    }));
    const manager = {
      getMessageIterator: async function* () {
        yield { type: 'summarize', last_user_message: 'done', last_assistant_message: 'ok', prompt_number: 1 };
      },
      confirmClaimedMessages: mock(() => Promise.resolve(1)),
      resetProcessingToPending: mock(() => Promise.resolve(0)),
    } as unknown as SessionManager;
    const provider = new StubbedCodexProvider(manager, ['']);

    await provider.startSession(makeSession());

    expect(provider.sentModels).not.toContain('claude-sonnet-4-6');
    expect(provider.sentModels.every((model) => model === 'gpt-5.4-mini')).toBe(true);
  });

  it('raises a transient failure when a clean turn carried no agent_message at all', () => {
    let thrown: unknown;
    try {
      parseCodexJsonl(CODEX_NO_ANSWER_JSONL);
    } catch (error: unknown) {
      thrown = error;
    }
    expect(isClassified(thrown)).toBe(true);
    expect(isClassified(thrown) ? thrown.kind : undefined).toBe('transient');
  });
});
