// The API key lives in sessionStorage: it survives a reload but not a closed
// tab, and unlike localStorage it is not shared with every other tab forever.
// Where storage is unavailable (some private modes), it lives in memory for
// this page instead, so a signed-in page still sends it.
const KEY = "incidentiq.apiKey";
let memory: string | null = null;

export function getApiKey(): string | null {
  try {
    return sessionStorage.getItem(KEY) ?? memory;
  } catch {
    return memory;
  }
}

export function setApiKey(value: string | null): void {
  memory = value;
  try {
    if (value) sessionStorage.setItem(KEY, value);
    else sessionStorage.removeItem(KEY);
  } catch {
    // Kept in memory above.
  }
}
