import { SseParser } from "./stream";

describe("SSE parser", () => {
  it("parses messages split across chunks, with CRLF line endings and pings", () => {
    const parser = new SseParser();
    // sse-starlette separates lines with \r\n and sends ": ping" comments.
    expect(parser.push("event: snapshot\r\ndata: {\"status\":")).toEqual([]);
    expect(parser.push(" \"running\"}\r\n\r\n: ping - 2026\r\n\r\nevent: end\r\n")).toEqual([
      { event: "snapshot", data: '{"status": "running"}' },
    ]);
    expect(parser.push("data: complete\r\n\r\n")).toEqual([{ event: "end", data: "complete" }]);
  });

  it("joins multi-line data and defaults the event name", () => {
    const parser = new SseParser();
    expect(parser.push("data: a\ndata: b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
  });
});

describe("SSE parser across chunk boundaries", () => {
  const frame = 'event: snapshot\r\ndata: {"status": "running"}\r\n\r\nevent: end\r\ndata: complete\r\n\r\n';

  it("gives the same messages wherever the network splits the frame", () => {
    const expected = [
      { event: "snapshot", data: '{"status": "running"}' },
      { event: "end", data: "complete" },
    ];
    for (let cut = 0; cut <= frame.length; cut++) {
      const parser = new SseParser();
      const got = [...parser.push(frame.slice(0, cut)), ...parser.push(frame.slice(cut))];
      expect(got, `split at ${cut}`).toEqual(expected);
    }
  });

  it("survives one character per chunk", () => {
    const parser = new SseParser();
    const got = [...frame].flatMap((ch) => parser.push(ch));
    expect(got.map((m) => m.event)).toEqual(["snapshot", "end"]);
  });
});
