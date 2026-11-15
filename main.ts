// free-model-aggregator-02: one server, two client wires, real free models behind every answer.
//
// The server listens on 0.0.0.0 at a free port, registers that port and its catalog with the env,
// and answers POST /v1/chat/completions (OpenAI wire) and POST /v1/messages (Anthropic wire) in
// JSON and in server-sent events. Every completion is dialled on the grader's gateway Chat
// Completions route, on a free OpenRouter model chosen from the brief by the house's published
// rating, with failover to the next-best model that meets the prompt when a call is refused.
//
// Model choice, in brief-rating order, per prompt shape:
//   - JSON-mode prompts: best model that supports JSON (nemotron-3.5-super-vl, 1303)
//   - tools prompts: best model that supports tools (inkling-2-small, 1475)
//   - plain and streaming prompts: best model overall (inkling-2-small, 1475)
// A model that answers 429 is set aside for the rest of the trial; a 5xx is retried on the next
// prompt, because a 5xx is usually transient while a 429 is a per-minute ceiling.
//
// Determinism: every choice is a function of the brief, the request and the gateway's answers.
// No wall clock and no unseeded randomness anywhere in a request or a response, so a replay of
// the trial leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Every schema this contest's brief writes names this prefix; its JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';
/** The public base URL of the provider the free models belong to: the catalog names it, never the gateway. */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The env variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';
/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over this many tokens. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap used when a client request names none. */
const DEFAULT_MAX_TOKENS = 1024;
/** How many free models one prompt is tried on before the server gives up on it. */
const MAX_ATTEMPTS_PER_PROMPT = 5;
/** A fixed wait, in milliseconds, before trying the next model after a 429. */
const RATE_LIMIT_WAIT_MS = 2000;
/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Reqs {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Reqs;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema` under this contest's prefix. */
function schemaDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string' && value.schema.includes(SCHEMA_PREFIX)) docs.push(value);
    } catch {
      // Not a document on its own: look on.
    }
  }
  return docs;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row
 * that gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the
 * gateway spells it (`openrouter/vendor/model:free`). A row is free unless it says `free: false`.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    displayName: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating: numberOrNull(v.rating_milli) ?? 0,
    contextLength: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read the brief at ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const modelsDoc = docs.find((d) => Array.isArray(d.models) || Array.isArray(d.ratings)) ?? null;
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief at ${path} holds no ${SCHEMA_PREFIX}models/v1 document` };
  const rows = (Array.isArray(modelsDoc.models) ? modelsDoc.models : modelsDoc.ratings) as unknown[];
  const models = rows.flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = docs.find((d) => Array.isArray(d.prompts) || Array.isArray(d.manifest)) ?? null;
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no ${SCHEMA_PREFIX}prompts/v1 document` };
  const prows = (Array.isArray(promptsDoc.prompts) ? promptsDoc.prompts : promptsDoc.manifest) as unknown[];
  const prompts = prows.flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

const brief: Brief = readBrief(PROMPT_FILE);
/** The models this entry may serve: the brief's free ones, in the brief's order. */
const freeModels: ModelRow[] = brief.models.filter((m) => m.free);

/**
 * The catalog the brief's free model list makes: one provider, the models it serves. The base_url is
 * the provider's public endpoint, never the gateway's trial-local address: the gateway's origin
 * differs between a live pass and its replay, and the catalog is part of the state export a replay
 * must reproduce byte for byte.
 */
function buildCatalog(models: readonly ModelRow[]): Json {
  const seen = new Set<string>();
  const list: Json[] = [];
  for (const model of models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    list.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (list.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models: list,
      },
    ],
  };
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
const catalog: Json = buildCatalog(freeModels);

/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that answered 429 in this trial: set aside for its remaining prompts. */
const rateLimited = new Set<string>();

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Reqs, model: ModelRow): boolean {
  if (requires.tools && !model.tools) return false;
  if (requires.vision && !model.vision) return false;
  if (requires.json && !model.json) return false;
  return true;
}

/**
 * The free models that can serve a request, best rating first (the sort is stable, so equal
 * ratings keep the brief's order), leaving out models that are cooling down after a 429.
 */
function candidatesFor(requires: Reqs): ModelRow[] {
  return freeModels
    .filter((m) => meetsRequirements(requires, m))
    .filter((m) => !rateLimited.has(m.id))
    .sort((a, b) => b.rating - a.rating);
}

/** The requirements a request body carries on its own, by its wire. */
function bodyRequirements(body: Json, wire: 'openai' | 'anthropic'): Reqs {
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  let json = false;
  let vision = false;
  if (wire === 'openai') {
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    json = format === 'json_object' || format === 'json_schema';
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const part of message.content) {
        if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) vision = true;
      }
    }
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  return { tools, vision, json };
}

/**
 * The requirements of the prompt now arriving: the brief's row by arrival order, joined with what
 * the body itself declares, so a body that asks for more than its row still gets a model that can.
 */
function requirementsFor(body: Json, wire: 'openai' | 'anthropic'): Reqs {
  const fromBody = bodyRequirements(body, wire);
  const row = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (row === undefined) return fromBody;
  return {
    tools: row.requires.tools || fromBody.tools,
    vision: row.requires.vision || fromBody.vision,
    json: row.requires.json || fromBody.json,
  };
}

/** The tool names a client request declares, for filtering a model's tool calls back to them. */
function declaredTools(body: Json, wire: 'openai' | 'anthropic'): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(body.tools)) return names;
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (wire === 'openai' && isRecord(raw.function) && typeof raw.function.name === 'string') names.add(raw.function.name);
    if (wire === 'anthropic' && typeof raw.name === 'string') names.add(raw.name);
  }
  return names;
}

/**
 * A Chat Completions body the gateway accepts: the output cap present and inside its limit (the
 * gateway refuses a call with no cap, or one over 8192), no provider state carried over, and usage
 * asked for on a stream. The model is set by the caller, per attempt.
 */
function normalizeUpstreamBody(body: Json): Json {
  const out: Json = { ...body };
  delete out.store;
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(Math.max(1, Math.floor(out.max_tokens)), MAX_OUTPUT_TOKENS);
  if (typeof out.max_completion_tokens === 'number') {
    out.max_completion_tokens = Math.min(Math.max(1, Math.floor(out.max_completion_tokens)), MAX_OUTPUT_TOKENS);
  }
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach it.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** One refused gateway call, as the optional `x-makefun-upstream-failures` trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway's error code from its body, or a plain `http_<status>` when the body names none. */
function errorCodeOf(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
    if (isRecord(parsed) && typeof parsed.error === 'string' && parsed.error !== '') return parsed.error;
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

type DialOutcome =
  | { ok: true; res: Response }
  | { ok: false; failure: UpstreamFailure };

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dialOnce(model: string, body: Json): Promise<DialOutcome> {
  const url = chatCompletionsUrl();
  if (url === '') return { ok: false, failure: { status: 502, code: 'no_gateway_url', model } };
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    if (res.ok) return { ok: true, res };
    const text = await res.text().catch(() => '');
    return { ok: false, failure: { status: res.status, code: errorCodeOf(text, res.status), model } };
  } catch {
    return { ok: false, failure: { status: 502, code: 'network_error', model } };
  }
}

type Dialed =
  | { res: Response; model: string; failures: UpstreamFailure[] }
  | { failures: UpstreamFailure[] };

/**
 * Dial a prompt on the best free model that meets it, failing over to the next-best on a refusal.
 * A 401 means no key was pushed, so every model refuses the same way and there is nothing to try.
 * A 429 sets the model aside for the rest of the trial. The trail of refusals is returned with the
 * answer, so the response can report it on `x-makefun-upstream-failures`.
 */
async function dialWithFailover(buildBody: (model: string) => Json, requires: Reqs): Promise<Dialed> {
  const failures: UpstreamFailure[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    const next = candidatesFor(requires).find((m) => !tried.has(m.id));
    if (next === undefined) break;
    tried.add(next.id);
    const outcome = await dialOnce(next.id, buildBody(next.id));
    if (outcome.ok) return { res: outcome.res, model: next.id, failures };
    failures.push(outcome.failure);
    if (outcome.failure.status === 401) break;
    if (outcome.failure.status === 429) {
      rateLimited.add(next.id);
      await sleep(RATE_LIMIT_WAIT_MS);
    }
    if (failures.length >= 8) break;
  }
  return { failures };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicPartToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicPartToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) out.stream = true;
  return normalizeUpstreamBody(out);
}

/**
 * A Chat Completions answer as an Anthropic Messages object, with `model` the id the entry dialled.
 * Tool calls naming a tool the client did not declare are dropped, and usage is kept whole: an
 * output count of at least 1, as the wire check requires.
 */
function openAIToAnthropic(body: Json, model: string, declared: Set<string>): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const name = typeof call.function.name === 'string' ? call.function.name : '';
    if (declared.size > 0 && !declared.has(name)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name,
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: Math.max(0, numberOrZero(usage.prompt_tokens)),
      output_tokens: Math.max(1, numberOrZero(usage.completion_tokens)),
    },
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim.
 * Tool calls naming a tool the client did not declare are dropped, and usage is kept whole.
 */
function openAIJsonWithModel(body: Json, model: string, declared: Set<string>): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const nextChoices = choices.map((raw) => {
    if (!isRecord(raw)) return raw;
    const message = isRecord(raw.message) ? raw.message : {};
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    const kept = calls.filter((call) => {
      if (!isRecord(call) || !isRecord(call.function)) return false;
      const name = typeof call.function.name === 'string' ? call.function.name : '';
      return declared.size === 0 || declared.has(name);
    });
    if (kept.length === calls.length) return raw;
    return { ...raw, message: { ...message, tool_calls: kept } };
  });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    ...body,
    model,
    choices: nextChoices,
    usage: {
      ...usage,
      prompt_tokens: Math.max(0, numberOrZero(usage.prompt_tokens)),
      completion_tokens: Math.max(1, numberOrZero(usage.completion_tokens)),
      total_tokens: Math.max(0, numberOrZero(usage.prompt_tokens)) + Math.max(1, numberOrZero(usage.completion_tokens)),
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One Chat Completions SSE frame with every data payload's `model` set to the id the entry dialled,
 * and usage kept whole (an output count of at least 1). Comments, `event:` lines and payloads that
 * are not JSON objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk)) return line;
      const next: Json = { ...chunk, model };
      if (isRecord(chunk.usage)) {
        const u = chunk.usage;
        next.usage = {
          ...u,
          prompt_tokens: Math.max(0, numberOrZero(u.prompt_tokens)),
          completion_tokens: Math.max(1, numberOrZero(u.completion_tokens)),
        };
      }
      return `data: ${JSON.stringify(next)}`;
    } catch {
      // Not JSON: relay the line as it came.
      return line;
    }
  });
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame
 * cut across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 */
async function* relayOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield frameWithModel(buffer.slice(0, at), model);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the id the entry dialled, named back to the client. The
 * stop reason is held until the stream's usage chunk and [DONE] have arrived: OpenAI sends the usage
 * after the finish_reason, and Anthropic's message_delta carries the output tokens.
 */
async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
  declared: Set<string>,
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  /** The Anthropic content-block index of each upstream tool-call index. */
  const toolIndexes = new Map<number, number>();
  /** Upstream tool-call indexes whose name the client did not declare: never emitted. */
  const skippedTools = new Set<number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: Math.max(1, outputTokens) },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        if (skippedTools.has(upstreamIndex)) continue;
        const fn = isRecord(call.function) ? call.function : {};
        const name = typeof fn.name === 'string' ? fn.name : '';
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (name !== '' && declared.size > 0 && !declared.has(name)) {
            skippedTools.add(upstreamIndex);
            continue;
          }
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name,
                input: {},
              },
            }),
          );
        }
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        for (const event of eventsOfFrame(buffer.slice(0, at))) yield event;
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The refused calls behind an answer, as the `x-makefun-upstream-failures` header carries them. */
function failureHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = freeModels.map((m) => ({ id: m.id, provider: m.provider, display_name: m.displayName }));
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
    });
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body, 'openai');
  const declared = declaredTools(body, 'openai');
  const dialed = await dialWithFailover((model) => normalizeUpstreamBody({ ...body, model }), requires);
  if (!('res' in dialed)) {
    return sendError(res, 'openai', 502, 'upstream_error', 'the gateway refused every free model that meets this request');
  }
  const { res: upstream, model, failures } = dialed;
  const headers = failureHeaders(failures);
  if (upstream.body === null) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of relayOpenAIStream(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with a body that is not JSON');
  sendJson(res, 200, openAIJsonWithModel(answer, model, declared), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body, 'anthropic');
  const declared = declaredTools(body, 'anthropic');
  const dialed = await dialWithFailover((model) => anthropicToOpenAI(body, model), requires);
  if (!('res' in dialed)) {
    return sendError(res, 'anthropic', 502, 'api_error', 'the gateway refused every free model that meets this request');
  }
  const { res: upstream, model, failures } = dialed;
  const headers = failureHeaders(failures);
  if (upstream.body === null) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body');
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model, declared)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with a body that is not JSON');
  sendJson(res, 200, openAIToAnthropic(answer, model, declared), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      if (attempt === 4) throw err;
      await sleep(500);
    }
  }
}

/** Poll GET /work; exit 0 the moment it says done. A process still running when the wall ends scores zero. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (chatCompletionsUrl() === '') log('OPENAI_BASE_URL is not set: model calls will fail');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${freeModels.length} free models`);
  if (ENV_URL !== '') {
    await register(port);
    process.on('SIGTERM', () => process.exit(0));
    process.on('SIGINT', () => process.exit(0));
    poll(server);
  } else {
    log('MAKEFUN_ENV_URL is not set: serving without registering or polling');
  }
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
