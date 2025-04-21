use core::arch::asm;
pub fn interpret_func() -> isize {
    0
}

#[no_mangle]
pub fn trap_handler(func_index: usize, ctx: *mut usize) {
    // todo

    let ret = interpret_func();
    // write return value to a0
    unsafe {
        *ctx.add(10) = ret;
    }
    trap_return(ctx);
}

pub fn trap_return(ctx: *const usize) {
    extern "C" {
        fn __ret();
    }
    asm!(
        "jr {ret_va}",
        ret_va = in(reg) __ret as usize,
        in("sp") ctx,
        options(noreturn)
    );
}