// bun_darwin_run_loop_attach: native code that started NSApplication on the main thread asks Bun
// to wait inside AppKit's run loop instead of kevent64 (src/runtime/darwin_run_loop.rs). The
// helper below plays that native code: it starts NSApplication without a window, and schedules
// work the way AppKit would run it inside the wait (a run-loop timer calling into JavaScript, a
// tracking loop like an open menu's). Runs in a subprocess so AppKit never loads in the runner.
// Flags: 1 keeps the process alive while attached (nothing else here holds it while a callback
// is awaited), 2 is LIVE_TURNS.
import { expect, test } from "bun:test";
import { bunEnv, bunExe, isMacOS, tempDir } from "harness";
import { join } from "path";

const cc = Bun.which("cc") || Bun.which("clang");

const helper = `
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <objc/objc-exception.h>
#include <objc/message.h>
#include <objc/runtime.h>

typedef void (*callback_t)(void);

static id send(id receiver, const char *selector) {
  return ((id (*)(id, SEL))objc_msgSend)(receiver, sel_registerName(selector));
}

// NSApplication on this (the main) thread, launched, with no window and no Dock icon.
int app_start(void) {
  if (!dlopen("/System/Library/Frameworks/AppKit.framework/AppKit", RTLD_LAZY)) return -1;
  id app = send((id)objc_getClass("NSApplication"), "sharedApplication");
  if (!app) return -2;
  ((void (*)(id, SEL, long))objc_msgSend)(app, sel_registerName("setActivationPolicy:"), 1);
  send(app, "finishLaunching");
  return 0;
}

static void call_out(CFRunLoopTimerRef timer, void *info) { ((callback_t)info)(); }

// Calls into JavaScript from a run-loop timer after \`seconds\`, from inside whatever waits then.
void call_later(double seconds, callback_t cb) {
  CFRunLoopTimerContext context = {0, (void *)cb, NULL, NULL, NULL};
  CFRunLoopTimerRef timer = CFRunLoopTimerCreate(NULL, CFAbsoluteTimeGetCurrent() + seconds, 0, 0, 0, call_out, &context);
  CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes);
  CFRelease(timer);
}

static double loop_seconds;
static int loop_kind;  // 0: an open menu's tracking loop, 1: native code waiting in a run-loop mode
static int loop_state; // 0 not yet, 1 running, 2 done
static double loop_ms;

static void run_loop_below(void *unused) {
  loop_state = 1;
  CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
  if (loop_kind == 0) {
    // What NSMenu and a live resize do: take only the events they track, in the tracking mode,
    // until done. Anything else (Bun's wake event included) stays queued.
    id app = send((id)objc_getClass("NSApplication"), "sharedApplication");
    id mode = (id)CFSTR("NSEventTrackingRunLoopMode");
    while (CFAbsoluteTimeGetCurrent() - start < loop_seconds) {
      id until = ((id (*)(id, SEL, double))objc_msgSend)((id)objc_getClass("NSDate"),
          sel_registerName("dateWithTimeIntervalSinceNow:"), loop_seconds - (CFAbsoluteTimeGetCurrent() - start));
      ((id (*)(id, SEL, unsigned long long, id, id, signed char))objc_msgSend)(app,
          sel_registerName("nextEventMatchingMask:untilDate:inMode:dequeue:"), 1ull << 2 /* left mouse up */, until, mode, 1);
    }
  } else {
    // Native code waiting on the run loop in another mode where Bun's sources also live (a modal
    // panel's): Bun's wakes must not stop it (posting an NSEvent stops the current run loop).
    CFRunLoopRunInMode(CFSTR("NSModalPanelRunLoopMode"), loop_seconds, false);
  }
  loop_ms = (CFAbsoluteTimeGetCurrent() - start) * 1000;
  loop_state = 2;
}

int loop_done(void) { return loop_state == 2; }
double loop_duration_ms(void) { return loop_ms; }

static void raise_now(CFRunLoopTimerRef timer, void *unused) {
  id exception = ((id (*)(id, SEL, id, id, id))objc_msgSend)((id)objc_getClass("NSException"),
      sel_registerName("exceptionWithName:reason:userInfo:"), (id)CFSTR("BunRunLoopTest"),
      (id)CFSTR("raised inside the run loop"), NULL);
  objc_exception_throw(exception);
}

// Raises an NSException \`after\` seconds from now from a run-loop timer, code AppKit runs while
// Bun waits in it. (Not from a dispatch block: libdispatch ends the process on an exception that
// escapes one, in any app.)
void raise_later(double after) {
  CFRunLoopTimerRef timer = CFRunLoopTimerCreate(NULL, CFAbsoluteTimeGetCurrent() + after, 0, 0, 0, raise_now, NULL);
  CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes);
  CFRelease(timer);
}

// Runs one of those loops for \`seconds\`, \`after\` seconds from now, from the main queue, so
// with no JavaScript on the stack.
void loop_later(int kind, double after, double seconds) {
  loop_kind = kind;
  loop_seconds = seconds;
  loop_state = 0;
  dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(after * 1e9)), dispatch_get_main_queue(), NULL, run_loop_below);
}
`;

const fixture = `
import { dlopen, FFIType, JSCallback } from "bun:ffi";
const native = dlopen(process.argv[2], {
  app_start: { args: [], returns: FFIType.i32 },
  call_later: { args: [FFIType.f64, FFIType.function], returns: FFIType.void },
  loop_later: { args: [FFIType.i32, FFIType.f64, FFIType.f64], returns: FFIType.void },
  loop_done: { args: [], returns: FFIType.i32 },
  loop_duration_ms: { args: [], returns: FFIType.f64 },
  raise_later: { args: [FFIType.f64], returns: FFIType.void },
}).symbols;
const bun = dlopen(process.execPath, {
  bun_darwin_run_loop_attach: { args: [FFIType.u32], returns: FFIType.i32 },
  bun_darwin_run_loop_detach: { args: [], returns: FFIType.void },
  bun_darwin_run_loop_stats: { args: [FFIType.ptr], returns: FFIType.void },
}).symbols;
const names = ["waits", "timedOut", "woken", "full", "dispatched", "staleWakes", "kqueueCallbacks", "deadlines", "nestedTurns", "lateTotalUs", "lateMaxUs", "exceptions"];
const stats = () => { const b = new BigUint64Array(names.length); bun.bun_darwin_run_loop_stats(b); return Object.fromEntries(names.map((n, i) => [n, Number(b[i])])); };
const out = {};

out.start = native.app_start();
out.attach = bun.bun_darwin_run_loop_attach(Number(process.argv[3]));

// Timers fire on time while Bun waits inside AppKit.
const late = [];
for (let i = 0; i < 10; i++) { const t = performance.now(); await Bun.sleep(20); late.push(performance.now() - t - 20); }
out.timerLateMax = Math.max(...late);
out.waitsAfterTimers = stats().waits;

// JavaScript called from inside the wait settles a promise: its reactions run before AppKit
// sleeps again, not at Bun's next turn (no timer is armed here to provide one).
{
  let called = 0, resolve;
  const cb = new JSCallback(() => { called = performance.now(); resolve(); }, { args: [], returns: "void" });
  const settled = new Promise(r => (resolve = r));
  native.call_later(0.05, cb.ptr);
  await settled;
  out.callbackToContinuationMs = performance.now() - called;
  cb.close();
}

// A loop below the wait for 400 ms, with a 10 ms interval running: how long the interval stalls,
// how many turns of Bun's loop ran inside, and how long the loop itself lasted.
async function below(kind) {
  let last = performance.now(), maxGap = 0;
  const tick = setInterval(() => { const n = performance.now(); maxGap = Math.max(maxGap, n - last); last = n; }, 10);
  const before = stats().nestedTurns;
  native.loop_later(kind, 0.05, 0.4);
  while (!native.loop_done()) await Bun.sleep(10);
  await Bun.sleep(30);
  clearInterval(tick);
  return { maxTimerGapMs: maxGap, nestedTurns: stats().nestedTurns - before, loopMs: native.loop_duration_ms() };
}
out.tracking = await below(0);
out.nativeWait = await below(1);

// An NSException raised by code AppKit runs while Bun waits is reported (logged, as
// -[NSApplication run] does) and the process goes on: timers still fire afterwards.
{
  const before = stats().exceptions;
  native.raise_later(0.05);
  while (stats().exceptions === before) await Bun.sleep(10);
  const t = performance.now();
  await Bun.sleep(20);
  out.exceptions = stats().exceptions - before;
  out.timerAfterExceptionMs = performance.now() - t;
}

bun.bun_darwin_run_loop_detach();
console.log(JSON.stringify(out));
`;

async function run(flags: number) {
  using dir = tempDir("darwin-run-loop", { "helper.c": helper, "fixture.mjs": fixture });
  const dylib = join(String(dir), "helper.dylib");
  await using compile = Bun.spawn({
    cmd: [cc!, "-dynamiclib", "-o", dylib, join(String(dir), "helper.c"), "-framework", "CoreFoundation", "-lobjc"],
    env: bunEnv,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [, ccErr, ccExit] = await Promise.all([compile.stdout.text(), compile.stderr.text(), compile.exited]);
  if (ccExit !== 0) console.error(ccErr);
  expect(ccExit).toBe(0);

  await using proc = Bun.spawn({
    cmd: [bunExe(), join(String(dir), "fixture.mjs"), dylib, String(flags)],
    env: bunEnv,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([proc.stdout.text(), proc.stderr.text(), proc.exited]);
  if (exitCode !== 0) console.error(stderr);
  expect(exitCode).toBe(0);
  // -[NSApplication reportException:] logs the exception's reason.
  expect(stderr).toContain("raised inside the run loop");
  return JSON.parse(stdout.trim().split("\n").at(-1)!);
}

test.skipIf(!isMacOS || !cc)("waits inside AppKit's run loop: timers on time, callout reactions at once", async () => {
  const out = await run(1);
  expect(out.start).toBe(0);
  expect(out.attach).toBe(0);
  expect(out.waitsAfterTimers).toBeGreaterThan(0);
  expect(out.timerLateMax).toBeLessThan(15);
  expect(out.callbackToContinuationMs).toBeLessThan(10);
  // Without LIVE_TURNS a tracking loop holds Bun's loop until it ends, and nothing ends it early.
  expect(out.tracking.nestedTurns).toBe(0);
  expect(out.tracking.maxTimerGapMs).toBeGreaterThan(300);
  expect(out.tracking.loopMs).toBeGreaterThan(380);
  // Bun's wakes do not cut short native code waiting in another run-loop mode.
  expect(out.nativeWait.loopMs).toBeGreaterThan(380);
  // An Objective-C exception inside AppKit's wait is reported and Bun keeps running.
  expect(out.exceptions).toBe(1);
  expect(out.timerAfterExceptionMs).toBeLessThan(40);
});

test.skipIf(!isMacOS || !cc)("LIVE_TURNS keeps timers running inside a tracking loop", async () => {
  const out = await run(3);
  expect(out.attach).toBe(0);
  expect(out.tracking.nestedTurns).toBeGreaterThan(0);
  expect(out.tracking.maxTimerGapMs).toBeLessThan(60);
  expect(out.tracking.loopMs).toBeGreaterThan(380);
  expect(out.nativeWait.loopMs).toBeGreaterThan(380);
});
