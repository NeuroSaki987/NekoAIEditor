# KV debugger recipe

1. Load `sshleifer/tiny-gpt2`.
2. Open KV Cache Debugger.
3. Prefill with a short prompt.
4. Inspect `layer=0`, `component=key`, `slice=:, 0, -1:, :`.
5. Apply a small edit, for example mode `add`, value `0.01`, strength `0.2`.
6. Queue next token.
7. Execute queued token.
8. Export KV state and import it again.
