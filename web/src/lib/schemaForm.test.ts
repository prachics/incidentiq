import type { JsonSchema } from "../api/types";
import { fieldsFromSchema, toArguments, toFormValues, validate } from "./schemaForm";

// The schema pydantic publishes for scale_service / restart_service arguments.
const SCALE: JsonSchema = {
  type: "object",
  properties: {
    service: { type: "string", title: "Service", description: "Exact service name." },
    replica_count: { type: "integer", minimum: 1, maximum: 50, title: "Replica Count" },
    instance_id: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Instance Id" },
    reason: { type: "string", minLength: 10, title: "Reason" },
  },
  required: ["service", "replica_count", "reason"],
};

describe("schema-driven argument form", () => {
  const fields = fieldsFromSchema(SCALE);

  it("reads kinds, bounds and nullability from the schema", () => {
    const byName = Object.fromEntries(fields.map((f) => [f.name, f]));
    expect(byName.replica_count).toMatchObject({ kind: "integer", minimum: 1, maximum: 50, required: true });
    expect(byName.instance_id).toMatchObject({ kind: "string", nullable: true, required: false });
    expect(byName.reason?.minLength).toBe(10);
  });

  it("accepts the agent's own arguments", () => {
    const values = toFormValues(fields, {
      service: "checkout-service",
      replica_count: 6,
      instance_id: null,
      reason: "pool saturation under peak load",
    });
    expect(validate(fields, values)).toEqual({});
  });

  it("rejects what the server would reject, before sending", () => {
    const errors = validate(fields, {
      service: "",
      replica_count: "500",
      instance_id: "",
      reason: "too short",
    });
    expect(errors).toEqual({
      service: "Required",
      replica_count: "Must be at most 50",
      reason: "At least 10 characters",
    });
    expect(validate(fields, { service: "x", replica_count: "2.5", reason: "long enough reason" })).toEqual({
      replica_count: "Must be a whole number",
    });
  });

  it("sends numbers as numbers and blank optional fields as null", () => {
    const args = toArguments(fields, {
      service: " checkout-service ",
      replica_count: "4",
      instance_id: "",
      reason: "smaller step first",
    });
    expect(args).toEqual({
      service: "checkout-service",
      replica_count: 4,
      instance_id: null,
      reason: "smaller step first",
    });
  });
});
