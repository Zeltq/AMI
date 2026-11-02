# Project instructions

## Background work updates

For data collection, model training, evaluation, and other long background tasks,
provide a concise status update at least every 10 minutes while waiting.
During active implementation, ordinary progress updates satisfy this rule.
Before yielding while background work remains, create or update a native Codex
thread heartbeat with a 10-minute interval; do not create duplicate monitors.
Verify the actual process is alive, then report progress, rate, errors/retries,
and relevant training/validation metrics. Report completion or failure separately.
Keep current process commands, artifact paths and stage in runs/pilot-status.json.
Do not start duplicate jobs. Once all authorized work is finished, report the
result and pause the heartbeat. Apply this rule to future project tasks too.

## Implementation and verification

State material assumptions before implementation; do not hide uncertainty.
Make the smallest correct change, avoid unrelated rewrites and unnecessary abstractions.
Run the narrowest relevant checks first and the full suite for broad changes.
Only claim checks that were actually run. Final reports must state changes,
verification and remaining risks or assumptions.

## Communication style

Avoid using slop words or phrases like "Bottom Line:" in conclusions, "delve," "foster," "leverage," "it's worth noting," "importantly," "Question? Answer." or "This isn't about X. It's about Y.", "genuinely" or hyphenated compound descriptions and adjectives. Do not use concluding summary statements such as "In short:..", "The simplest mental model is:...".

State the intended action directly. Avoid adding what you won't do, what will remain unchanged, or how you'll separate or categorize results. Do not use contrastive framing such as "X, not Y" that introduces an unprompted alternative that the user didn't ask about. Avoid invented compound labels like "exact-head checks" and "editorial-row layouts", vague qualifiers, and canned transitions; use plain verbs and prepositions to state the actual relationship directly.
