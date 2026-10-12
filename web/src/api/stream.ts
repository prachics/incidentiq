import { authHeaders } from "./client";

export interface SseMessage {
  event: string;
  data: string;
}

/**
 * Incremental parser for a server-sent-events body. Feed it text chunks in the
 * order they arrive; it returns the complete messages so far and keeps any
 * partial message for the next chunk. Handles \n and \r\n line endings and
 * ignores comment lines (the server's keep-alive pings).
 */
export class SseParser {
  private buffer = "";
  // A chunk can end between the \r and \n of one line ending; the \r is held
  // until the next chunk shows whether it was CRLF or a lone CR.
  private pendingCr = false;

  push(chunk: string): SseMessage[] {
    let text = (this.pendingCr ? "\r" : "") + chunk;
    this.pendingCr = text.endsWith("\r");
    if (this.pendingCr) text = text.slice(0, -1);
    this.buffer += text.replace(/\r\n?/g, "\n");
    const messages: SseMessage[] = [];
    let end = this.buffer.indexOf("\n\n");
    while (end !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const message = parseBlock(block);
      if (message) messages.push(message);
      end = this.buffer.indexOf("\n\n");
    }
    return messages;
  }
}

function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/**
 * Read a server-sent-events endpoint with fetch, so the API key can travel in a
 * header (the browser's EventSource cannot send one). Resolves when the server
 * closes the stream; rejects on a non-2xx response or a network error. Whether
 * the stream ended properly (an `end` event) is for the caller to judge.
 */
export async function readStream(
  path: string,
  onMessage: (message: SseMessage) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(path, {
    headers: { Accept: "text/event-stream", ...authHeaders() },
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`stream failed: HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    for (const message of parser.push(decoder.decode(value, { stream: true }))) onMessage(message);
  }
}
