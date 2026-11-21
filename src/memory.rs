//! Host RAM: what the machine has, what it can give a new allocation now, and the cap on what this process owns outright.
//!
//! The cap bounds memory the process owns outright (baked tensors, LoRA factors, parked conversation states). The
//! mmapped GGUF is page cache the OS can evict, so a model larger than RAM still loads and streams. `available_ram` is
//! the number to size anything *new* from — on a box already serving a model it is far below `total_ram`, and budgeting
//! against the machine's size instead of its free space over-commits it (see `deepseek::mat::plan`).

/// Total physical RAM in bytes, or `None` if undetectable.
pub fn total_ram() -> Option<u64> { os_memory().0 }

/// Bytes the OS would give a new allocation now: what is free plus the page cache it can reclaim without swapping. It
/// counts this process's own mapped model as reclaimable, which is why a device budget sized from it still leaves room
/// for the file pages an offloaded expert would otherwise need a second copy of. None if undetectable (an old kernel
/// with no `MemAvailable`).
pub fn available_ram() -> Option<u64> { os_memory().1 }

/// (total, available) in bytes. Two names for one query, so neither is read from a different snapshot than the other.
#[cfg(target_os = "linux")]
fn os_memory() -> (Option<u64>, Option<u64>) {
    let m = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kb = |key: &str| m.lines().find_map(|l| l.strip_prefix(key)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()).map(|k| k << 10);
    (kb("MemTotal:"), kb("MemAvailable:"))
}

/// (total, available) in bytes. `ullAvailPhys` is what a new allocation can get without paging; `ullTotalPhys` the machine has.
#[cfg(windows)]
fn os_memory() -> (Option<u64>, Option<u64>) {
    #[repr(C)]
    struct MemStatus {
        len: u32,
        load: u32,
        total: u64,
        avail: u64,
        rest: [u64; 5],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
    }
    let mut m = MemStatus { len: size_of::<MemStatus>() as u32, load: 0, total: 0, avail: 0, rest: [0; 5] };
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 { return (None, None); }
    (if m.total > 0 { Some(m.total) } else { None }, if m.avail > 0 { Some(m.avail) } else { None })
}

#[cfg(not(any(target_os = "linux", windows)))]
fn os_memory() -> (Option<u64>, Option<u64>) { (None, None) }

/// Usable cap in bytes: `max_ram` if nonzero, else `total - margin`. `None` means unlimited (RAM undetectable).
pub fn ram_cap(margin: u64, max_ram: u64) -> Option<u64> {
    if max_ram > 0 {
        return Some(max_ram);
    }
    Some(total_ram()?.saturating_sub(margin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_max_ram_wins() {
        assert_eq!(ram_cap(512 << 20, 8 << 30), Some(8 << 30));
    }

    #[test]
    fn margin_is_subtracted() {
        if let Some(t) = total_ram() {
            assert_eq!(ram_cap(1 << 30, 0), Some(t.saturating_sub(1 << 30)));
        }
    }

    /// The two numbers come from one read, and free is never more than the machine's size.
    #[test]
    fn available_is_within_total() {
        let (t, a) = os_memory();
        if let (Some(t), Some(a)) = (t, a) {
            assert!(a <= t, "{a} available of {t} total");
            assert!(a > 0);
        }
    }
}
