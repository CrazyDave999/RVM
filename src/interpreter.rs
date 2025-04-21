use super::mem::KERNEL_INFO;
use core::arch::asm;
pub fn interpret_func() -> isize {
    0
}

#[no_mangle]
pub fn trap_handler(func_index: usize, ctx: *mut usize) {
    // todo

    let ret = interpret_func();
    trap_return(ret, ctx);
}

pub fn trap_return(ret: isize, ctx: *const usize) {
    extern "C" {
        fn __ret();
    }

    // update kernel stack pointer
    let mut sp: usize;
    unsafe {
        asm!(
            "mv {sp}, sp",
            sp = out(reg) sp,
            options(nomem, nostack)
        );

        KERNEL_INFO[0] = sp;

        asm!(
            "jr {ret_va}",
            ret_va = in(reg) __ret as usize,
            in("a0") ret,
            in("sp") ctx,
            options(noreturn, nomem, nostack)
        );
    }
}
