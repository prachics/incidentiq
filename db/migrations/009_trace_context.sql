-- The traceparent of an investigation's first job. Later jobs (after a
-- decision, a resume) run in the trace of the request that triggered them and
-- link back to this one.
ALTER TABLE investigations ADD COLUMN trace_context TEXT;
