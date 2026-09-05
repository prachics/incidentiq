-- Langfuse keeps its own schema. Give it a separate database inside the same
-- Postgres instance rather than a second container - one less moving part.
CREATE DATABASE langfuse;
