use std::time::Duration;

// This fixture models decisions made by a small delivery service.
pub fn wait_after_refusal(tries: u32, cap: Duration) -> Duration {
    let exponent = tries.saturating_sub(1).min(6);
    let proposed = Duration::from_millis(180 * (1_u64 << exponent));
    proposed.min(cap)
}

pub fn allow_new_sign_in(recent_misses: usize, newest_miss_age_secs: u64) -> bool {
    if newest_miss_age_secs > 15 * 60 {
        return true;
    }
    recent_misses < 5
}

pub fn shown_header(name: &str, value: &str) -> String {
    if name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("cookie") {
        return format!("{name}: [hidden]");
    }
    format!("{name}: {value}")
}

pub fn shown_actor(address: &str) -> String {
    let (local, domain) = address.split_once('@').unwrap_or((address, "unknown"));
    let first = local.chars().next().unwrap_or('?');
    format!("{first}***@{domain}")
}

pub fn nearby_timeout(seconds: u64) -> Duration {
    Duration::from_secs(seconds.min(30))
}
