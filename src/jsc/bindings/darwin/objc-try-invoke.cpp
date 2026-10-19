// The catch frames around Objective-C in bun: a message sent under try, so an
// exception raised inside AppKit while Bun waits in its run loop
// (src/runtime/darwin_run_loop.rs) comes back as a value instead of ending
// the process. Plain C++ so that bun links no Objective-C runtime and carries
// no Objective-C image info: the caller loads the runtime itself and hands
// these frames the entry points they need. An Objective-C exception is a C++
// exception whose type_info sits inside the thrown buffer right after the
// `id` (objc4's `struct objc_exception`), which is how the object is
// recovered here. From oven-sh/bun#39979 (Alistair Smith).

#include <cxxabi.h>
#include <typeinfo>

using Send = void (*)(void* receiver, void* selector, void* argument);
using NextEvent = void* (*)(void* app, void* selector, unsigned long long mask, void* until, void* mode, signed char dequeue);
using Retain = void* (*)(void* object);

// Release builds strip __TEXT,__unwind_info, and ld64.lld drops a frame's
// DWARF FDE whenever it can encode the frame compactly, which would leave a
// catch frame nowhere. Compact unwind cannot describe CFI it does not model,
// so these two no-op directives make the compiler mark the enclosing frame
// DWARF-only on every architecture: the FDE stays in __eh_frame and
// libunwind finds it there.
#define KEEP_DWARF_UNWIND_INFO() __asm__ volatile(".cfi_remember_state\n\t.cfi_restore_state")

/// Inside a `catch (...)` handler: whether what was thrown is an
/// Objective-C object, and which (nil included).
static inline __attribute__((always_inline)) bool thrownObject(void** object)
{
    const std::type_info* type = abi::__cxa_current_exception_type();
    void* thrown = abi::__cxa_current_primary_exception();
    bool objc = type && thrown && reinterpret_cast<const char*>(type) == static_cast<const char*>(thrown) + sizeof(void*);
    *object = objc ? *static_cast<void**>(thrown) : nullptr;
    if (thrown)
        abi::__cxa_decrement_exception_refcount(thrown);
    return objc;
}

/// Inside a `catch (...)` handler: what an Objective-C `@throw` threw
/// (usually an NSException), with a +1 reference from `retain` that the
/// caller releases, or nullptr when `nil` was thrown. Anything else thrown
/// keeps unwinding.
static inline __attribute__((always_inline)) void* caughtObject(Retain retain)
{
    void* object;
    if (!thrownObject(&object))
        throw;
    return object ? retain(object) : nullptr;
}

/// [receiver selector:argument] through `msgSend`. On an Objective-C
/// exception, stores what was thrown in *exception (see caughtObject) and
/// returns false.
extern "C" bool Bun__objc__trySend(Send msgSend, Retain retain, void* receiver, void* selector, void* argument, void** exception)
{
    KEEP_DWARF_UNWIND_INFO();
    try {
        msgSend(receiver, selector, argument);
        return true;
    } catch (...) {
        *exception = caughtObject(retain);
        return false;
    }
}

/// [app nextEventMatchingMask:until:inMode:dequeue:] through `msgSend`, the
/// event in *event. Timers, observers and sources AppKit runs while it waits
/// are other code's; on an Objective-C exception raised by one, stores what
/// was thrown in *exception and returns false.
extern "C" bool Bun__objc__tryNextEvent(NextEvent msgSend, Retain retain, void* app, void* selector, unsigned long long mask, void* until, void* mode, signed char dequeue, void** event, void** exception)
{
    KEEP_DWARF_UNWIND_INFO();
    try {
        *event = msgSend(app, selector, mask, until, mode, dequeue);
        return true;
    } catch (...) {
        *exception = caughtObject(retain);
        return false;
    }
}
