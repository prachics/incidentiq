import type { Field, FormValues } from "../lib/schemaForm";

interface Props {
  fields: Field[];
  values: FormValues;
  errors: Record<string, string>;
  onChange: (values: FormValues) => void;
}

const INPUT =
  "w-full rounded-md border px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-slate-400";

export function SchemaForm({ fields, values, errors, onChange }: Props) {
  const set = (name: string, value: string | boolean) => onChange({ ...values, [name]: value });
  return (
    <div className="space-y-3">
      {fields.map((f) => {
        const id = `arg-${f.name}`;
        const error = errors[f.name];
        const border = error ? "border-red-400" : "border-slate-300";
        const value = values[f.name];
        return (
          <div key={f.name}>
            <label htmlFor={id} className="block text-xs font-medium text-slate-700">
              {f.label}
              {f.required && !f.nullable && <span className="text-red-600"> *</span>}
            </label>
            {f.kind === "boolean" ? (
              <input
                id={id}
                type="checkbox"
                checked={Boolean(value)}
                onChange={(e) => set(f.name, e.target.checked)}
                className="mt-1"
              />
            ) : f.kind === "enum" ? (
              <select
                id={id}
                value={String(value ?? "")}
                onChange={(e) => set(f.name, e.target.value)}
                className={`${INPUT} ${border} mt-1 bg-white`}
              >
                <option value="">—</option>
                {f.options?.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            ) : f.name === "reason" || (f.minLength ?? 0) >= 10 ? (
              <textarea
                id={id}
                rows={2}
                value={String(value ?? "")}
                onChange={(e) => set(f.name, e.target.value)}
                className={`${INPUT} ${border} mt-1`}
              />
            ) : (
              <input
                id={id}
                type={f.kind === "string" ? "text" : "number"}
                min={f.minimum}
                max={f.maximum}
                value={String(value ?? "")}
                onChange={(e) => set(f.name, e.target.value)}
                className={`${INPUT} ${border} mt-1 font-mono`}
              />
            )}
            {error ? (
              <p className="mt-1 text-xs text-red-700">{error}</p>
            ) : (
              f.description && <p className="mt-1 text-xs text-slate-500">{f.description}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
