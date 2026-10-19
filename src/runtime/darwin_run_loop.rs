//! macOS: Bun's idle wait inside the AppKit run loop of the main thread.
//!
//! For a process whose native code (an FFI library, a N-API addon, SDL, GLFW)
//! has started `NSApplication` on the main thread, the thread Bun's event loop
//! runs on. Without this, AppKit only gets the thread when JavaScript pumps it
//! on a timer: idle CPU, latency equal to the pump interval, and windows that
//! stop answering while the loop sleeps in `kevent64`.
//!
//! Bun's event loop stays the outer loop. Once attached, the loop's idle wait
//! happens inside `-[NSApplication nextEventMatchingMask:…]` instead of
//! `kevent64`: the kqueue descriptor is a `CFFileDescriptor` run-loop source,
//! so the wait ends as soon as the kqueue has something (I/O, cross-thread
//! wakeups), a UI event arrives, or the `untilDate:` set to Bun's next timer
//! passes. UI events are dispatched right there with `sendEvent:`; then
//! control returns to the loop, which harvests the kqueue without blocking.
//! See `park_cb` in usockets' `epoll_kqueue.c`.
//!
//! JavaScript also runs *inside* that wait (an FFI callback from a delegate,
//! a block, a display timer). A run-loop observer asks before every sleep
//! when Bun next needs a turn ([`next_due`]) and arms a deadline that
//! ends the wait by posting an event, so a `setTimeout` armed there still
//! ends the wait on time. While AppKit runs a tracking loop of its own inside
//! `sendEvent:` (an open menu, a live resize) the wake event just queues;
//! Bun's timers and I/O resume when the gesture ends, unless the embedder
//! asked for [`LIVE_TURNS`]: then a kqueue wakeup or a due timer inside such a
//! loop runs one non-blocking turn of Bun's loop right there ([`nested_turn`]),
//! as Electron runs libuv inside Chromium's message loop, so timers, network
//! and IPC keep going while a menu is open. Never with JavaScript on the stack
//! (a menu opened from JavaScript is a synchronous call, and stays one).
//!
//! Because all of the above runs JavaScript, the loop does not release JS heap
//! access or hand this thread's allocator heaps to the scavenger around
//! `park_cb` the way it does around `kevent64`; two run-loop observers hand
//! the heaps over around the run loop's own sleep instead ([`hand_off`],
//! [`take_back`]), in every mode.
//!
//! The deadline is a kernel timer (`EVFILT_TIMER`, `NOTE_CRITICAL`) on a
//! private kqueue that is a second `CFFileDescriptor` source, not a
//! `CFRunLoopTimer` or the wait's `untilDate:`: those are coalesced by the
//! system (measured: 100 ms and more past a 50 ms deadline for a process
//! started outside the GUI session), while Bun's own `kevent64` timeouts are
//! not. Every timed wait arms it, so Bun's timers fire on time inside AppKit.
//!
//! The run-loop half of oven-sh/bun#39979 (`src/appkit/run_loop.rs` there,
//! by Alistair Smith), without the Objective-C bridge: Objective-C and
//! CoreFoundation are reached through `dlsym`, so the binary links nothing
//! new, and nothing here runs until native code calls
//! [`bun_darwin_run_loop_attach`].

use core::cell::{Cell, RefCell};
use core::ffi::{CStr, c_char, c_int, c_uint, c_void};
use core::mem::transmute;
use core::ptr;

use bun_core::Timespec;
use bun_io::KeepAlive;
use bun_jsc::virtual_machine::VirtualMachine;
use bun_mimalloc_sys::mimalloc::{mi_on_thread_idle, mi_on_thread_idle_end, mi_on_thread_idle_start};
use bun_uws_sys::Loop;

type Id = *mut c_void;
type Sel = *mut c_void;

/// `NSEventTypeApplicationDefined`.
const APPLICATION_DEFINED: usize = 15;
/// `NSEventMaskAny`.
const ANY_EVENT: u64 = u64::MAX;
/// Marks the application-defined event we post to end a wait.
const WAKE_SUBTYPE: i16 = 0x4275;
/// `kCFFileDescriptorReadCallBack`.
const READ_CALLBACK: usize = 1;
/// `kCFRunLoopBeforeWaiting`.
const BEFORE_WAITING: usize = 1 << 5;
/// `kCFRunLoopAfterWaiting`.
const AFTER_WAITING: usize = 1 << 6;
/// How often the owner thread sweeps its own heaps before a sleep when there
/// is no scavenger to hand them to (the rate usockets uses).
const IDLE_SWEEP_INTERVAL_NS: u64 = 100_000_000;
/// Past CoreAnimation's commit observer (2 000 000), so JavaScript run by a
/// display callback in the same pass is seen before the loop sleeps.
const OBSERVER_ORDER: isize = 3_000_000;
/// How often a busy (non-idle) tick still drains UI events.
const BUSY_DRAIN_INTERVAL_NS: u64 = 1_000_000;
/// UI events dispatched per park before Bun's loop gets a turn; the rest stay
/// queued and end the next wait at once.
const EVENTS_PER_PARK: u32 = 16;
/// A `CFAbsoluteTime` further off than any deadline (CF clamps it).
const NEVER: f64 = 1.0e12;
/// `EVFILT_TIMER` flags: `data` in nanoseconds, and no coalescing leeway.
const NOTE_NSECONDS: u32 = 0x0000_0004;
const NOTE_CRITICAL: u32 = 0x0000_0020;
/// Seconds past Bun's deadline that the wait's own `untilDate:` ends it, if
/// the deadline event has not.
const UNTIL_BACKSTOP: f64 = 0.25;
/// Seconds; see [`Park::arm_within`].
const REARM_SLACK: f64 = 0.000_25;
/// Seconds between fresh wake events while a posted one has not come back to
/// `pump` (a nested loop that takes every event may have consumed it).
const WAKE_RECHECK: Timespec = Timespec {
    sec: 0,
    nsec: 100_000_000,
};

#[repr(C)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
struct CFContext {
    version: isize,
    info: *mut c_void,
    retain: *const c_void,
    release: *const c_void,
    copy_description: *const c_void,
}

type FdCallback = unsafe extern "C" fn(*mut c_void, usize, *mut c_void);
type ObserverCallback = unsafe extern "C" fn(*mut c_void, usize, *mut c_void);

/// The Objective-C runtime and CoreFoundation entry points, from `dlsym`.
#[allow(non_snake_case)]
struct Rt {
    msg_send: *const c_void,
    sel_register_name: unsafe extern "C" fn(*const c_char) -> Sel,
    get_class: unsafe extern "C" fn(*const c_char) -> Id,
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
    CFRunLoopGetMain: unsafe extern "C" fn() -> *mut c_void,
    CFFileDescriptorCreate:
        unsafe extern "C" fn(*const c_void, c_int, u8, FdCallback, *const CFContext) -> *mut c_void,
    CFFileDescriptorCreateRunLoopSource:
        unsafe extern "C" fn(*const c_void, *mut c_void, isize) -> *mut c_void,
    CFFileDescriptorEnableCallBacks: unsafe extern "C" fn(*mut c_void, usize),
    CFRunLoopAddSource: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void),
    CFRunLoopObserverCreate: unsafe extern "C" fn(
        *const c_void,
        usize,
        u8,
        isize,
        ObserverCallback,
        *const c_void,
    ) -> *mut c_void,
    CFRunLoopAddObserver: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void),
    CFAbsoluteTimeGetCurrent: unsafe extern "C" fn() -> f64,
    CFRunLoopCopyCurrentMode: unsafe extern "C" fn(*mut c_void) -> *const c_void,
    CFEqual: unsafe extern "C" fn(*const c_void, *const c_void) -> u8,
    CFRelease: unsafe extern "C" fn(*const c_void),
    common_modes: *const c_void,
    default_mode: *const c_void,
}

fn sym(handle: *mut c_void, name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `handle` is a live dlopen handle and `name` is NUL-terminated.
    let p = unsafe { libc::dlsym(handle, name.as_ptr()) };
    (!p.is_null()).then_some(p)
}

fn open(path: &CStr) -> Option<*mut c_void> {
    // SAFETY: NUL-terminated path; the handle is kept for the process.
    let h = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
    (!h.is_null()).then_some(h)
}

impl Rt {
    fn load() -> Option<Rt> {
        let objc = open(c"/usr/lib/libobjc.A.dylib")?;
        let cf = open(c"/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")?;
        macro_rules! f {
            ($h:expr, $n:literal) => {
                // SAFETY: the symbol has the signature of the field it fills.
                unsafe { transmute(sym($h, $n)?) }
            };
        }
        // The two modes are `const CFStringRef` globals: dlsym gives their address.
        // SAFETY: both symbols are initialized CFStringRef variables in CoreFoundation.
        let common_modes = unsafe { *(sym(cf, c"kCFRunLoopCommonModes")? as *const *const c_void) };
        // SAFETY: as above.
        let default_mode = unsafe { *(sym(cf, c"kCFRunLoopDefaultMode")? as *const *const c_void) };
        Some(Rt {
            msg_send: sym(objc, c"objc_msgSend")?,
            sel_register_name: f!(objc, c"sel_registerName"),
            get_class: f!(objc, c"objc_getClass"),
            pool_push: f!(objc, c"objc_autoreleasePoolPush"),
            pool_pop: f!(objc, c"objc_autoreleasePoolPop"),
            CFRunLoopGetMain: f!(cf, c"CFRunLoopGetMain"),
            CFFileDescriptorCreate: f!(cf, c"CFFileDescriptorCreate"),
            CFFileDescriptorCreateRunLoopSource: f!(cf, c"CFFileDescriptorCreateRunLoopSource"),
            CFFileDescriptorEnableCallBacks: f!(cf, c"CFFileDescriptorEnableCallBacks"),
            CFRunLoopAddSource: f!(cf, c"CFRunLoopAddSource"),
            CFRunLoopObserverCreate: f!(cf, c"CFRunLoopObserverCreate"),
            CFRunLoopAddObserver: f!(cf, c"CFRunLoopAddObserver"),
            CFAbsoluteTimeGetCurrent: f!(cf, c"CFAbsoluteTimeGetCurrent"),
            CFRunLoopCopyCurrentMode: f!(cf, c"CFRunLoopCopyCurrentMode"),
            CFEqual: f!(cf, c"CFEqual"),
            CFRelease: f!(cf, c"CFRelease"),
            common_modes,
            default_mode,
        })
    }

    fn sel(&self, name: &CStr) -> Sel {
        // SAFETY: NUL-terminated selector name.
        unsafe { (self.sel_register_name)(name.as_ptr()) }
    }

    /// `objc_msgSend` cast to the method's C signature `F`.
    ///
    /// # Safety
    /// `F` must be the `extern "C"` signature of the method sent, receiver and
    /// selector first.
    unsafe fn send<F: Copy>(&self) -> F {
        debug_assert_eq!(size_of::<F>(), size_of::<*const c_void>());
        // SAFETY: per the caller; objc_msgSend takes any method's signature on
        // arm64 and, for methods that return no struct, on x86_64.
        unsafe { core::mem::transmute_copy(&self.msg_send) }
    }

    fn pool_push(&self) -> *mut c_void {
        // SAFETY: no preconditions.
        unsafe { (self.pool_push)() }
    }

    fn pool_pop(&self, pool: *mut c_void) {
        // SAFETY: `pool` came from `pool_push` on this thread and is the innermost one.
        unsafe { (self.pool_pop)(pool) }
    }
}

/// The selectors and classes the pump sends to, resolved once.
struct Sels {
    next_event: Sel,
    send_event: Sel,
    update_windows: Sel,
    post_event: Sel,
    other_event: Sel,
    kind: Sel,
    subtype: Sel,
    date_with_time: Sel,
    ns_event: Id,
    ns_date: Id,
}

struct Park {
    rt: Rt,
    sels: Sels,
    app: Id,
    /// The loop whose idle wait comes through here; the VM's, so it outlives us.
    loop_: *mut Loop,
    kqueue_fd: i32,
    /// `CFFileDescriptorRef` over the loop's kqueue; lives for the process.
    fd_source: *mut c_void,
    /// Private kqueue holding the deadline timer, and its `CFFileDescriptor`
    /// source in the common modes. Armed by `wait` for Bun's next timer and by
    /// `before_waiting` for a deadline that came up inside a wait.
    deadline_kq: i32,
    deadline_source: *mut c_void,
    distant_past: Id,
    distant_future: Id,
    attached: Cell<bool>,
    /// [`LIVE_TURNS`] was asked for.
    live: Cell<bool>,
    /// Inside [`nested_turn`].
    in_turn: Cell<bool>,
    last_drain_ns: Cell<u64>,
    /// A wake asked for while the run loop was in a mode of someone else's:
    /// posted once it is back in ours (see [`wake`]).
    pending_wake: Cell<bool>,
    /// Set by `wake` so `pump` can tell its own wake event from a stale one.
    woke: Cell<bool>,
    /// Inside `pump`: AppKit may be running JavaScript below us.
    in_pump: Cell<bool>,
    /// What the deadline is armed for, as a `CFAbsoluteTime`; `NEVER` when idle.
    armed: Cell<f64>,
    /// The `untilDate:` of the wait in progress; `NEVER` outside a wait.
    until: Cell<f64>,
    /// The heaps are the scavenger's until `take_back` runs.
    handed_off: Cell<bool>,
    last_sweep_ns: Cell<u64>,
    /// Autorelease pool left in place between two outermost parks, so what
    /// native code autoreleases while Bun runs timers and I/O callbacks is
    /// freed at the next park (the pool `-[NSApplication run]` wraps around
    /// each event). Null when none.
    tick_pool: Cell<*mut c_void>,
    /// Scoped pools of ours currently pushed below a park.
    scoped_pools: Cell<u32>,
    keep_alive: RefCell<KeepAlive>,
    stats: Cell<Stats>,
}

/// Counters for tests and benchmarks of the pump, read with
/// [`bun_darwin_run_loop_stats`]. Times in microseconds.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Stats {
    /// Idle waits begun in AppKit.
    waits: u64,
    /// Waits that ended at their `untilDate:` (Bun's next timer).
    timed_out: u64,
    /// Waits ended by our wake event.
    woken: u64,
    /// Waits ended after the event budget.
    full: u64,
    /// Events sent to the application.
    dispatched: u64,
    /// Wake events skipped as stale.
    stale_wakes: u64,
    /// kqueue run-loop source callbacks.
    kqueue_callbacks: u64,
    /// Deadline timer callbacks.
    deadlines: u64,
    /// Turns of Bun's loop run inside a tracking loop.
    nested_turns: u64,
    /// Total and worst time a timed-out wait ended past its `untilDate:`.
    late_total_us: u64,
    late_max_us: u64,
}

impl Park {
    fn count(&self, f: impl FnOnce(&mut Stats)) {
        let mut stats = self.stats.get();
        f(&mut stats);
        self.stats.set(stats);
    }
}

/// Copies the counters so far into `out` (zeroes before the first attach).
#[unsafe(no_mangle)]
unsafe extern "C" fn bun_darwin_run_loop_stats(out: *mut u64) {
    let stats = park().map(|p| p.stats.get()).unwrap_or_default();
    // SAFETY: the caller passes room for every field of `Stats`.
    unsafe { out.cast::<Stats>().write_unaligned(stats) };
}

thread_local! {
    static PARK: Cell<*const Park> = const { Cell::new(ptr::null()) };
}

fn park() -> Option<&'static Park> {
    let p = PARK.get();
    // SAFETY: set once in `attach` to a leaked allocation.
    (!p.is_null()).then(|| unsafe { &*p })
}

/// Flags of [`bun_darwin_run_loop_attach`]: keep the process running while attached.
const KEEP_ALIVE: c_uint = 1;
/// Flags of [`bun_darwin_run_loop_attach`]: run Bun's loop inside AppKit's tracking and modal loops.
const LIVE_TURNS: c_uint = 2;

/// Result codes of [`bun_darwin_run_loop_attach`].
const OK: c_int = 0;
const NOT_MAIN_THREAD: c_int = -1;
const NO_APPKIT: c_int = -2;

/// Routes the idle wait of Bun's event loop through the AppKit run loop of
/// the main thread, from now on. For native code that has created
/// `NSApplication` (and called `finishLaunching`) on the main thread, inside
/// a Bun process. `flags`: [`KEEP_ALIVE`] (1) keeps the process running while
/// attached, as an open window would; [`LIVE_TURNS`] (2) keeps timers, I/O and
/// IPC running while AppKit tracks a menu or a resize. A later call replaces
/// the flags. Returns 0, -1 off the main thread or without a VM there, -2 when
/// AppKit is not loaded.
#[unsafe(no_mangle)]
extern "C" fn bun_darwin_run_loop_attach(flags: c_uint) -> c_int {
    // SAFETY: no preconditions.
    if unsafe { libc::pthread_main_np() } != 1 {
        return NOT_MAIN_THREAD;
    }
    let Some(vm) = VirtualMachine::get_or_null() else {
        return NOT_MAIN_THREAD;
    };
    // SAFETY: the VM installed on this thread.
    let vm = unsafe { &*vm };
    if park().is_none() {
        let Some(new) = install(vm.uws_loop_mut()) else {
            return NO_APPKIT;
        };
        PARK.set(Box::leak(Box::new(new)));
    }
    let park = park().expect("installed");
    park.attached.set(true);
    park.live.set(flags & LIVE_TURNS != 0);
    vm.uws_loop_mut().internal_loop_data.park_cb = Some(park_cb);
    if flags & KEEP_ALIVE != 0 {
        park.keep_alive.borrow_mut().ref_(bun_io::js_vm_ctx());
    } else {
        park.keep_alive.borrow_mut().unref(bun_io::js_vm_ctx());
    }
    OK
}

/// Puts the idle wait back in `kevent64` and drops the keep-alive. The
/// run-loop sources stay installed but do nothing. Main thread only.
#[unsafe(no_mangle)]
extern "C" fn bun_darwin_run_loop_detach() {
    let Some(park) = park() else { return };
    // SAFETY: no preconditions.
    if unsafe { libc::pthread_main_np() } != 1 {
        return;
    }
    park.attached.set(false);
    // SAFETY: `loop_` is the VM's loop, alive for the rest of the process.
    unsafe { (*park.loop_).internal_loop_data.park_cb = None };
    park.keep_alive.borrow_mut().unref(bun_io::js_vm_ctx());
}

fn install(loop_: &mut Loop) -> Option<Park> {
    let rt = Rt::load()?;
    // SAFETY: NUL-terminated class names.
    let (app_class, ns_event, ns_date) = unsafe {
        (
            (rt.get_class)(c"NSApplication".as_ptr()),
            (rt.get_class)(c"NSEvent".as_ptr()),
            (rt.get_class)(c"NSDate".as_ptr()),
        )
    };
    if app_class.is_null() || ns_event.is_null() || ns_date.is_null() {
        return None;
    }
    let sels = Sels {
        next_event: rt.sel(c"nextEventMatchingMask:untilDate:inMode:dequeue:"),
        send_event: rt.sel(c"sendEvent:"),
        update_windows: rt.sel(c"updateWindows"),
        post_event: rt.sel(c"postEvent:atStart:"),
        other_event: rt.sel(
            c"otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:",
        ),
        kind: rt.sel(c"type"),
        subtype: rt.sel(c"subtype"),
        date_with_time: rt.sel(c"dateWithTimeIntervalSinceReferenceDate:"),
        ns_event,
        ns_date,
    };
    // SAFETY: class methods with no arguments returning an object; the
    // shared application and the two dates live for the process.
    let (app, distant_past, distant_future) = unsafe {
        let send0: unsafe extern "C" fn(Id, Sel) -> Id = rt.send();
        (
            send0(app_class, rt.sel(c"sharedApplication")),
            send0(ns_date, rt.sel(c"distantPast")),
            send0(ns_date, rt.sel(c"distantFuture")),
        )
    };
    if app.is_null() {
        return None;
    }
    let kqueue_fd = loop_.fd;
    let context = CFContext {
        version: 0,
        info: ptr::null_mut(),
        retain: ptr::null(),
        release: ptr::null(),
        copy_description: ptr::null(),
    };
    // SAFETY: CF calls with valid arguments. The descriptor is not closed on
    // invalidate (Bun owns it), and the CF objects are intentionally never
    // released.
    let (fd_source, timer) = unsafe {
        let main = (rt.CFRunLoopGetMain)();
        let fd_source = (rt.CFFileDescriptorCreate)(ptr::null(), kqueue_fd, 0, on_kqueue_readable, &raw const context);
        assert!(!fd_source.is_null(), "CFFileDescriptorCreate");
        let source = (rt.CFFileDescriptorCreateRunLoopSource)(ptr::null(), fd_source, 0);
        assert!(!source.is_null(), "CFFileDescriptorCreateRunLoopSource");
        (rt.CFRunLoopAddSource)(main, source, rt.common_modes);
        let deadline_kq = libc::kqueue();
        assert!(deadline_kq >= 0, "kqueue");
        libc::fcntl(deadline_kq, libc::F_SETFD, libc::FD_CLOEXEC);
        let deadline_source = (rt.CFFileDescriptorCreate)(ptr::null(), deadline_kq, 0, on_deadline, &raw const context);
        assert!(!deadline_source.is_null(), "CFFileDescriptorCreate");
        let source = (rt.CFFileDescriptorCreateRunLoopSource)(ptr::null(), deadline_source, 0);
        assert!(!source.is_null(), "CFFileDescriptorCreateRunLoopSource");
        (rt.CFRunLoopAddSource)(main, source, rt.common_modes);
        // Last before the sleep and first after it, so nothing runs on this
        // thread between the two.
        for (activity, order, callback) in [
            (BEFORE_WAITING, OBSERVER_ORDER, before_waiting as ObserverCallback),
            (BEFORE_WAITING, isize::MAX, hand_off as ObserverCallback),
            (AFTER_WAITING, isize::MIN, take_back as ObserverCallback),
        ] {
            let observer = (rt.CFRunLoopObserverCreate)(ptr::null(), activity, 1, order, callback, ptr::null());
            assert!(!observer.is_null(), "CFRunLoopObserverCreate");
            (rt.CFRunLoopAddObserver)(main, observer, rt.common_modes);
        }
        (fd_source, (deadline_kq, deadline_source))
    };
    let (deadline_kq, deadline_source) = timer;
    Some(Park {
        rt,
        sels,
        app,
        loop_: ptr::from_mut(loop_),
        kqueue_fd,
        fd_source,
        deadline_kq,
        deadline_source,
        distant_past,
        distant_future,
        attached: Cell::new(false),
        live: Cell::new(false),
        in_turn: Cell::new(false),
        last_drain_ns: Cell::new(0),
        woke: Cell::new(false),
        pending_wake: Cell::new(false),
        in_pump: Cell::new(false),
        armed: Cell::new(NEVER),
        until: Cell::new(NEVER),
        handed_off: Cell::new(false),
        last_sweep_ns: Cell::new(0),
        tick_pool: Cell::new(ptr::null_mut()),
        scoped_pools: Cell::new(0),
        keep_alive: RefCell::new(KeepAlive::init()),
        stats: Cell::new(Stats::default()),
    })
}

/// How long until Bun's loop next needs a turn: zero when tasks or
/// immediates are queued, otherwise the time to the earliest armed timer
/// (either heap) or QUIC tick, or `None`. Peeks only; runs nothing.
fn next_due() -> Option<Timespec> {
    let vm = VirtualMachine::get();
    let event_loop = vm.event_loop_mut();
    let has_pending = !event_loop.immediate_tasks.is_empty()
        || !event_loop.next_immediate_tasks.is_empty()
        || event_loop.has_pending_tasks();
    let quic_next_tick_us = {
        let ild = &vm.uws_loop_mut().internal_loop_data;
        (!ild.quic_head.is_null()).then_some(ild.quic_next_tick_us)
    };
    crate::jsc_hooks::timer_all_mut().peek_next_due(has_pending, quic_next_tick_us)
}

/// No JavaScript frame is on the stack (so no native code JavaScript called
/// into can be holding an autorelease pool across this park), and the loop
/// is not inside `EventLoop::enter`.
fn outermost() -> bool {
    let vm = VirtualMachine::get();
    !vm.jsc_vm().is_entered() && vm.event_loop_mut().entered_event_loop_count == 0
}

/// Posts the application-defined event that makes a blocked
/// `nextEventMatchingMask:` return; once per wait, however many sources ask.
///
/// Not while the run loop runs in a mode other than ours: posting an event
/// stops the current run loop, which cuts short a `CFRunLoopRunInMode` that
/// native code is waiting in (and a menu's tracking loop would only queue the
/// event anyway). The wake is then posted by `before_waiting` once the run
/// loop is back in our mode.
fn wake() {
    let Some(park) = park() else { return };
    if !park.attached.get() || park.woke.get() {
        return;
    }
    if park.in_pump.get() && !current_mode_is_default(park) {
        park.pending_wake.set(true);
        return;
    }
    park.woke.set(true);
    // SAFETY: the signatures of `+[NSEvent otherEventWithType:…]` and
    // `-[NSApplication postEvent:atStart:]`.
    unsafe {
        let other: unsafe extern "C" fn(Id, Sel, usize, CGPoint, usize, f64, isize, Id, i16, isize, isize) -> Id =
            park.rt.send();
        let event = other(
            park.sels.ns_event,
            park.sels.other_event,
            APPLICATION_DEFINED,
            CGPoint { x: 0.0, y: 0.0 },
            0,
            0.0,
            0,
            ptr::null_mut(),
            WAKE_SUBTYPE,
            0,
            0,
        );
        if !event.is_null() {
            let post: unsafe extern "C" fn(Id, Sel, Id, i8) = park.rt.send();
            post(park.app, park.sels.post_event, event, 1);
        }
    }
}

impl Park {
    fn tick_depth(&self) -> i32 {
        // SAFETY: `loop_` is the VM's loop, alive for the rest of the process.
        unsafe { (*self.loop_).internal_loop_data.tick_depth }
    }

    fn now(&self) -> f64 {
        // SAFETY: no preconditions.
        unsafe { (self.rt.CFAbsoluteTimeGetCurrent)() }
    }

    /// Arms the deadline for `at`, a `CFAbsoluteTime` (`NEVER` disarms).
    fn arm(&self, at: f64) {
        if self.armed.replace(at) == at {
            return;
        }
        let mut kev: libc::kevent = unsafe { core::mem::zeroed() };
        kev.ident = 1;
        kev.filter = libc::EVFILT_TIMER;
        if at >= NEVER {
            kev.flags = libc::EV_DELETE;
        } else {
            kev.flags = libc::EV_ADD | libc::EV_ONESHOT;
            kev.fflags = NOTE_NSECONDS | NOTE_CRITICAL;
            kev.data = ((at - self.now()).max(0.0) * 1e9) as isize;
        }
        // SAFETY: one valid change on our private kqueue; no events out. A
        // delete of a timer that already fired fails with ENOENT, harmlessly.
        unsafe { libc::kevent(self.deadline_kq, &raw const kev, 1, ptr::null_mut(), 0, ptr::null()) };
        if at < NEVER {
            // SAFETY: the live CFFileDescriptor created in `install`.
            unsafe { (self.rt.CFFileDescriptorEnableCallBacks)(self.deadline_source, READ_CALLBACK) };
        }
    }

    /// Brings the deadline in to `timeout` from now if that is sooner than
    /// both the wait's own `untilDate:` and what the timer is armed for.
    fn arm_within(&self, timeout: &Timespec) {
        let at = self.now() + seconds(timeout);
        if at + REARM_SLACK < self.armed.get().min(self.until.get()) {
            self.arm(at);
        }
    }

    fn enable_fd_callback(&self) {
        // SAFETY: `fd_source` is the live CFFileDescriptor created in `install`.
        unsafe { (self.rt.CFFileDescriptorEnableCallBacks)(self.fd_source, READ_CALLBACK) };
    }

    fn date(&self, at: f64) -> Id {
        // SAFETY: `+[NSDate dateWithTimeIntervalSinceReferenceDate:]`; autoreleased
        // into the pool `park_cb` holds.
        unsafe {
            let f: unsafe extern "C" fn(Id, Sel, f64) -> Id = self.rt.send();
            f(self.sels.ns_date, self.sels.date_with_time, at)
        }
    }
}

/// Ends the wait; inside a tracking loop the wake sits in the queue until the
/// gesture ends.
unsafe extern "C" fn on_kqueue_readable(_f: *mut c_void, _types: usize, _info: *mut c_void) {
    if let Some(park) = park() {
        park.count(|s| s.kqueue_callbacks += 1);
    }
    if let Some(park) = park()
        && in_foreign_loop(park)
    {
        nested_turn(park);
        return;
    }
    wake();
}

unsafe extern "C" fn on_deadline(_f: *mut c_void, _types: usize, _info: *mut c_void) {
    if let Some(park) = park() {
        // Consume the fired timer so the private kqueue reads empty again.
        let mut kev: libc::kevent = unsafe { core::mem::zeroed() };
        let zero = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: room for one event, no wait.
        unsafe { libc::kevent(park.deadline_kq, ptr::null(), 0, &raw mut kev, 1, &raw const zero) };
        park.armed.set(NEVER);
        park.count(|s| s.deadlines += 1);
        if in_foreign_loop(park) {
            nested_turn(park);
            return;
        }
        if park.in_pump.get() {
            // A wake posted earlier that `pump` has still not seen is either
            // queued behind a tracking loop's narrower mask or was consumed by
            // a nested loop that takes every event; post afresh either way.
            park.woke.set(false);
            wake();
        }
    }
}

/// Runs before the run loop sleeps, in every mode. JavaScript that ran since
/// the wait began may have queued a task, an immediate or an earlier timer
/// than the one the wait was armed for; ask, and bring the deadline in.
unsafe extern "C" fn before_waiting(_o: *mut c_void, _activity: usize, _info: *mut c_void) {
    let Some(park) = park() else { return };
    if !park.in_pump.get() {
        return;
    }
    microtask_checkpoint();
    if !current_mode_is_default(park) {
        if !in_foreign_loop(park) {
            return;
        }
        // A tracking loop begun from a busy tick's drain has the kqueue
        // callback still off; nothing is left unharvested by a watch armed now.
        park.enable_fd_callback();
        // Our wait's `untilDate:` does not end a tracking loop: only the
        // deadline timer can, so arm it for Bun's next due alone.
        if let Some(due) = next_due() {
            let at = park.now() + seconds(&due);
            if at + REARM_SLACK < park.armed.get() {
                park.arm(at);
            }
        }
        return;
    }
    if park.pending_wake.replace(false) {
        wake();
        return;
    }
    if park.woke.get() {
        park.arm_within(&WAKE_RECHECK);
        return;
    }
    if let Some(due) = next_due() {
        park.arm_within(&due);
    }
}

/// JavaScript that native code ran inside the wait (an FFI callback from a
/// delegate, a block) settles promises; their reactions run here, before the
/// run loop sleeps, as they would after a task, and not at Bun's next turn.
/// Not with JavaScript on the stack (a synchronous menu opened from it).
fn microtask_checkpoint() {
    let vm = VirtualMachine::get();
    let event_loop = vm.event_loop_mut();
    if vm.jsc_vm().is_entered() || event_loop.entered_event_loop_count != 0 {
        return;
    }
    // The outermost exit is the checkpoint.
    event_loop.enter();
    event_loop.exit();
}

/// Runs last before the run loop sleeps, in every mode: this thread's heaps
/// go to the scavenger for the sleep, as they do around `kevent64`. With no
/// scavenger they are swept here, rate limited.
unsafe extern "C" fn hand_off(_o: *mut c_void, _activity: usize, _info: *mut c_void) {
    let Some(park) = park() else { return };
    if !park.attached.get() {
        return;
    }
    // SAFETY: the run loop sleeps next and `take_back` is the first thing it
    // runs after; nothing allocates on this thread in between.
    if unsafe { mi_on_thread_idle_start() } {
        park.handed_off.set(true);
        return;
    }
    let now = monotonic_ns();
    if now.wrapping_sub(park.last_sweep_ns.get()) >= IDLE_SWEEP_INTERVAL_NS {
        park.last_sweep_ns.set(now);
        mi_on_thread_idle();
    }
}

/// Runs first after the run loop wakes: the heaps come back before anything
/// can allocate.
unsafe extern "C" fn take_back(_o: *mut c_void, _activity: usize, _info: *mut c_void) {
    let Some(park) = park() else { return };
    if park.handed_off.replace(false) {
        mi_on_thread_idle_end();
    }
}

unsafe extern "C" fn park_cb(_loop: *mut Loop, timeout: *const Timespec) {
    let Some(park) = park() else { return };
    // SAFETY: timeout is null (forever) or points at the caller's timespec.
    let timeout = unsafe { timeout.as_ref() };
    let zero = timeout.is_some_and(|t| t.sec == 0 && t.nsec == 0);
    if park.in_pump.get() {
        // JavaScript that AppKit is running is waiting synchronously. Pumping
        // again here would dispatch events re-entrantly and could swallow the
        // outer wait's wake, so wait on the kqueue alone.
        if !zero {
            block_on_kqueue(park.kqueue_fd, timeout);
        }
        return;
    }
    // At the outermost tick, with no scoped pool of ours below, the previous
    // tick's pool is popped and a fresh one left in place until the next park.
    let scoped = if park.scoped_pools.get() == 0 && park.tick_depth() == 1 && outermost() {
        let old = park.tick_pool.replace(ptr::null_mut());
        if !old.is_null() {
            park.rt.pool_pop(old);
        }
        park.tick_pool.set(park.rt.pool_push());
        ptr::null_mut()
    } else {
        park.scoped_pools.set(park.scoped_pools.get() + 1);
        park.rt.pool_push()
    };
    // I/O already waiting on the kqueue makes this a busy tick too.
    if zero || kqueue_readable(park.kqueue_fd) {
        drain(park);
    } else {
        wait(park, timeout);
    }
    if !scoped.is_null() {
        park.rt.pool_pop(scoped);
        park.scoped_pools.set(park.scoped_pools.get() - 1);
    }
}

/// Whether AppKit is running a loop of its own below our wait (a menu's or a
/// live resize's tracking loop, a modal session) with no JavaScript on the
/// stack, so that a turn of Bun's loop can run here.
fn in_foreign_loop(park: &Park) -> bool {
    if !park.live.get() || !park.attached.get() || !park.in_pump.get() || park.in_turn.get() {
        return false;
    }
    !current_mode_is_default(park) && !VirtualMachine::get().jsc_vm().is_entered()
}

/// Whether the main run loop runs in the default mode, the one our wait uses.
fn current_mode_is_default(park: &Park) -> bool {
    // SAFETY: CF calls on the main run loop; the copied mode is released.
    unsafe {
        let mode = (park.rt.CFRunLoopCopyCurrentMode)((park.rt.CFRunLoopGetMain)());
        if mode.is_null() {
            return true;
        }
        let default = (park.rt.CFEqual)(mode, park.rt.default_mode) != 0;
        (park.rt.CFRelease)(mode);
        default
    }
}

/// One non-blocking turn of Bun's loop from inside a loop AppKit runs below
/// our wait: tasks and microtasks, immediates, I/O already ready (harvested
/// without blocking: the nested tick's `park_cb` sees `in_pump` and returns),
/// due timers. The loop's `tick_depth` defers socket frees to the outer tick.
fn nested_turn(park: &Park) {
    park.in_turn.set(true);
    park.count(|s| s.nested_turns += 1);
    let vm = VirtualMachine::get_mut_ptr();
    // SAFETY: the live per-thread VM, its event loop and uws loop; no
    // JavaScript is on the stack (`in_foreign_loop`), as at the top of a turn.
    unsafe {
        let el: *mut bun_jsc::event_loop::EventLoop = (*vm).event_loop;
        (*el).tick();
        (*el).tick_immediate_tasks(vm);
        (*(*el).usockets_loop()).tick_without_idle();
        let state = crate::jsc_hooks::runtime_state();
        if !state.is_null() {
            crate::timer::All::drain_timers(&mut (*state).timer, vm.cast());
        }
        (*el).tick();
        (*vm).on_after_event_loop();
        let _ = (*(*vm).global).handle_rejected_promises();
    }
    park.in_turn.set(false);
    // The kqueue was harvested: watch it again (CF callbacks are one-shot).
    park.enable_fd_callback();
}

/// Blocks until the kqueue has something or `timeout` (None = forever)
/// passes, without dequeuing anything.
fn block_on_kqueue(fd: i32, timeout: Option<&Timespec>) {
    let ms = timeout.map_or(-1, |t| {
        t.sec
            .saturating_mul(1000)
            .saturating_add((t.nsec + 999_999) / 1_000_000)
            .clamp(0, i64::from(i32::MAX)) as i32
    });
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd. Readiness, timeout and EINTR all just return.
    unsafe { libc::poll(&raw mut pfd, 1, ms) };
}

/// Whether the kqueue has events to harvest right now.
fn kqueue_readable(fd: i32) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd and no wait.
    unsafe { libc::poll(&raw mut pfd, 1, 0) > 0 && pfd.revents & libc::POLLIN != 0 }
}

/// Non-blocking: dispatch whatever UI events are already queued. Rate limited
/// so a loop that never idles pays ~2µs per millisecond rather than per tick.
fn drain(park: &Park) {
    if monotonic_ns().wrapping_sub(park.last_drain_ns.get()) >= BUSY_DRAIN_INTERVAL_NS {
        let _ = pump(park, park.distant_past);
        park.last_drain_ns.set(monotonic_ns());
    }
}

/// Blocks in AppKit until `timeout` (None = forever), a UI event, or kqueue
/// readiness; dispatches UI events that arrived.
fn wait(park: &Park, timeout: Option<&Timespec>) {
    // The kqueue was harvested after the previous park returned, so enabling
    // the callback here preserves harvest-then-arm.
    park.woke.set(false);
    park.pending_wake.set(false);
    park.enable_fd_callback();
    park.count(|s| s.waits += 1);
    let until = timeout.map(|t| park.now() + seconds(t));
    park.until.set(until.unwrap_or(NEVER));
    if let Some(at) = until {
        park.arm(at);
    }
    // The deadline ends the wait; `untilDate:` is only the backstop.
    let end = match until {
        Some(at) => pump(park, park.date(at + UNTIL_BACKSTOP)),
        None => pump(park, park.distant_future),
    };
    match end {
        PumpEnd::TimedOut => {
            let late_us = until.map_or(0, |at| ((park.now() - at).max(0.0) * 1e6) as u64);
            park.count(|s| {
                s.timed_out += 1;
                s.late_total_us += late_us;
                s.late_max_us = s.late_max_us.max(late_us);
            });
        }
        PumpEnd::Woken => park.count(|s| s.woken += 1),
        PumpEnd::Full => park.count(|s| s.full += 1),
    }
    park.until.set(NEVER);
    park.arm(NEVER);
    park.last_drain_ns.set(monotonic_ns());
}

fn seconds(t: &Timespec) -> f64 {
    t.sec as f64 + t.nsec as f64 / 1e9
}

/// Runs `nextEventMatchingMask:` until `deadline` for the first event and
/// without blocking for up to [`EVENTS_PER_PARK`] more, sending each event to
/// the application. Stops early at our wake event.
fn pump(park: &Park, deadline: Id) -> PumpEnd {
    let was_in_pump = park.in_pump.replace(true);
    // SAFETY: the C signatures of the methods sent.
    let (next_event, send_id, send0, get_usize, get_i16) = unsafe {
        let a: unsafe extern "C" fn(Id, Sel, u64, Id, *const c_void, i8) -> Id = park.rt.send();
        let b: unsafe extern "C" fn(Id, Sel, Id) = park.rt.send();
        let c: unsafe extern "C" fn(Id, Sel) = park.rt.send();
        let d: unsafe extern "C" fn(Id, Sel) -> usize = park.rt.send();
        let e: unsafe extern "C" fn(Id, Sel) -> i16 = park.rt.send();
        (a, b, c, d, e)
    };
    let s = &park.sels;
    let mut deadline = deadline;
    let mut dispatched = 0u32;
    let mut end = PumpEnd::TimedOut;
    loop {
        // SAFETY: `-[NSApplication nextEventMatchingMask:untilDate:inMode:dequeue:]`.
        let event = unsafe { next_event(park.app, s.next_event, ANY_EVENT, deadline, park.rt.default_mode, 1) };
        deadline = park.distant_past;
        if event.is_null() {
            break;
        }
        if dispatched == 0 {
            // An event came first: what ends the wait is no longer the deadline.
            end = PumpEnd::Woken;
        }
        // SAFETY: `-[NSEvent type]` and `-[NSEvent subtype]` (the latter only
        // for application-defined events, where it is valid).
        if unsafe { get_usize(event, s.kind) } == APPLICATION_DEFINED
            && unsafe { get_i16(event, s.subtype) } == WAKE_SUBTYPE
        {
            if park.woke.replace(false) {
                end = PumpEnd::Woken;
                break;
            }
            // A wake left over from an earlier park.
            park.count(|s| s.stale_wakes += 1);
            continue;
        }
        // SAFETY: `-[NSApplication sendEvent:]`.
        unsafe { send_id(park.app, s.send_event, event) };
        park.count(|s| s.dispatched += 1);
        dispatched += 1;
        if dispatched == EVENTS_PER_PARK {
            end = PumpEnd::Full;
            break;
        }
    }
    if dispatched > 0 {
        // SAFETY: `-[NSApplication updateWindows]`.
        unsafe { send0(park.app, s.update_windows) };
    }
    park.in_pump.set(was_in_pump);
    end
}

/// Why [`pump`] returned.
enum PumpEnd {
    /// `nextEventMatchingMask:` returned nil: the deadline passed (or, for a
    /// drain, nothing was queued).
    TimedOut,
    /// Our wake event, or UI events, ended it.
    Woken,
    /// [`EVENTS_PER_PARK`] events were dispatched.
    Full,
}

fn monotonic_ns() -> u64 {
    thread_local! { static START: std::time::Instant = std::time::Instant::now(); }
    START.with(|s| s.elapsed().as_nanos() as u64)
}
