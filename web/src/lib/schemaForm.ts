import type { JsonSchema } from "../api/types";

// Turns the JSON Schema the server publishes for a tool's arguments into form
// fields, and validates values against the same rules before they are sent.
// The server validates again; this only saves an approver a round trip.

export type FieldKind = "string" | "integer" | "number" | "boolean" | "enum";

export interface Field {
  name: string;
  label: string;
  kind: FieldKind;
  required: boolean;
  nullable: boolean;
  description?: string;
  options?: string[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
}

export type FormValues = Record<string, string | boolean>;

function concrete(schema: JsonSchema): { schema: JsonSchema; nullable: boolean } {
  if (!schema.anyOf) return { schema, nullable: false };
  const branches = schema.anyOf.filter((b) => b.type !== "null");
  return {
    schema: { ...schema, ...(branches[0] ?? { type: "string" }), anyOf: undefined },
    nullable: branches.length < schema.anyOf.length,
  };
}

function humanize(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function fieldsFromSchema(schema: JsonSchema | null | undefined): Field[] {
  const required = new Set(schema?.required ?? []);
  return Object.entries(schema?.properties ?? {}).map(([name, raw]) => {
    const { schema: s, nullable } = concrete(raw);
    const kind: FieldKind = s.enum
      ? "enum"
      : s.type === "integer" || s.type === "number" || s.type === "boolean"
        ? s.type
        : "string";
    return {
      name,
      label: s.title ?? raw.title ?? humanize(name),
      kind,
      required: required.has(name),
      nullable,
      description: raw.description ?? s.description,
      options: s.enum?.map(String),
      minimum: s.minimum,
      maximum: s.maximum,
      minLength: s.minLength,
      maxLength: s.maxLength,
    };
  });
}

/** Initial form values from an existing arguments object. */
export function toFormValues(fields: Field[], args: Record<string, unknown>): FormValues {
  const values: FormValues = {};
  for (const f of fields) {
    const v = args[f.name];
    values[f.name] = f.kind === "boolean" ? Boolean(v) : v === null || v === undefined ? "" : String(v);
  }
  return values;
}

/** Field name -> message. Empty when the values are valid. */
export function validate(fields: Field[], values: FormValues): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of fields) {
    const v = values[f.name];
    if (f.kind === "boolean") continue;
    const text = typeof v === "string" ? v.trim() : "";
    if (!text) {
      if (f.required && !f.nullable) errors[f.name] = "Required";
      continue;
    }
    if (f.kind === "integer" || f.kind === "number") {
      const n = Number(text);
      if (!Number.isFinite(n) || (f.kind === "integer" && !Number.isInteger(n))) {
        errors[f.name] = f.kind === "integer" ? "Must be a whole number" : "Must be a number";
      } else if (f.minimum !== undefined && n < f.minimum) {
        errors[f.name] = `Must be at least ${f.minimum}`;
      } else if (f.maximum !== undefined && n > f.maximum) {
        errors[f.name] = `Must be at most ${f.maximum}`;
      }
    } else if (f.kind === "enum" && f.options && !f.options.includes(text)) {
      errors[f.name] = `Must be one of ${f.options.join(", ")}`;
    } else if (f.minLength !== undefined && text.length < f.minLength) {
      errors[f.name] = `At least ${f.minLength} characters`;
    } else if (f.maxLength !== undefined && text.length > f.maxLength) {
      errors[f.name] = `At most ${f.maxLength} characters`;
    }
  }
  return errors;
}

/** Form values -> the JSON the API expects (numbers as numbers, blanks as null). */
export function toArguments(fields: Field[], values: FormValues): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const f of fields) {
    const v = values[f.name];
    if (f.kind === "boolean") {
      args[f.name] = Boolean(v);
      continue;
    }
    const text = typeof v === "string" ? v.trim() : "";
    if (!text) {
      if (f.nullable) args[f.name] = null;
      continue;
    }
    args[f.name] = f.kind === "integer" || f.kind === "number" ? Number(text) : text;
  }
  return args;
}
