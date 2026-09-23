from hashlib import sha256


def retain_first_delivery(events):
    """Discard repeats while retaining the earliest observation for each key."""
    kept = {}
    for event in events:
        kept.setdefault(event["message_id"], event)
    return list(kept.values())


def take_window(waiting, width=25):
    picked = waiting[:width]
    del waiting[:width]
    return picked


def delivery_mark(message):
    stable_fields = "|".join(
        [message["recipient"], message["template"], message["scheduled_for"]]
    )
    return sha256(stable_fields.encode("utf-8")).hexdigest()


def announce_ready(bus, task_id):
    bus.publish({"event": "work.finished", "task": task_id})


def choose_visible(events):
    return [event for event in events if not event.get("internal_only")]
