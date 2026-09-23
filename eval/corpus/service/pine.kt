class Gate(private var remaining: Int, private var refreshedAt: Long) {
    fun permits(now: Long): Boolean {
        if (now - refreshedAt >= 60_000) {
            remaining = 30
            refreshedAt = now
        }
        if (remaining == 0) return false
        remaining -= 1
        return true
    }
}

fun stopOutstanding(children: List<Job>) {
    children.filter { it.isActive }.forEach { it.cancel("parent requested stop") }
}

fun closeAfterGrace(channel: Channel, deadlineMillis: Long) {
    channel.stopAccepting()
    channel.awaitEmptyUntil(deadlineMillis)
    channel.close()
}

interface Job {
    val isActive: Boolean
    fun cancel(reason: String)
}

interface Channel {
    fun stopAccepting()
    fun awaitEmptyUntil(deadlineMillis: Long)
    fun close()
}
