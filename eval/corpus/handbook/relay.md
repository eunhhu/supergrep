# Relay handbook

## Checking incoming callbacks

Calculate the HMAC over the untouched request body and compare it with the
signature header using a constant-time comparison. Reject a callback with a
missing timestamp or one older than five minutes.

## Writing release notes

Group changes under Added, Changed, Fixed, and Removed. Mention a migration
step before the behavior change it requires, and link the issue identifier.

## Tone

Use short sentences when documenting operational steps.
