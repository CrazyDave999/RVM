.altmacro
.macro SAVE_GP n
	sd x\n, \n*8(sp)
.endm
.macro LOAD_GP n
    ld x\n, \n*8(sp)
.endm

.extern FUNC_TABLE
.extern asm_call_fn_handler
.extern alloc_stack
.extern dealloc_stack

	.global __asm_call_fn
	.global __interpreter_ret_asm
	.align 2
# from asm to interpreter/asm
# fn __asm_call_fn() -> i64;
__asm_call_fn:
	# now t0 is the func index, ra is next pc of the call inst

	# check if the target function has been compiled
	slli t1, t0, 3 # offset
	ld t1, FUNC_TABLE(t1) # load the func address
	bnez t1, __asm_call_asm

__asm_call_interpreter:
	# if not compiled, let the interpreter handle it
	# save callee-saved registers
	addi sp, sp, -8*32
	sd ra, 0*8(sp) # next pc

	.set n, 8
	.rept 2
		SAVE_GP %n
		.set n, n+1
	.endr

	# a0 and a1 may change after call alloc_stack, so save them here
	sd t0, 10*8(sp) # func index
	sd sp, 11*8(sp) # ctx

	.set n, 18
	.rept 10
		SAVE_GP %n
		.set n, n+1
	.endr

	addi t2, sp, 8*32
	sd t2, 2*8(sp) # old sp

	# switch stack
	call alloc_stack
	addi sp, a0, 1024*16 # new stack top

	# pub extern "C" fn asm_call_fn_handler(func_index: usize, ctx: *mut usize)
	ld a0, 10*8(sp) # func index
	ld a1, 11*8(sp) # ctx

	j asm_call_fn_handler

__asm_call_asm:
	# if compiled, simply jump
	jr t1

__interpreter_ret_asm:
	# now a0 is the return value, a1 is the ctx

	# going to call dealloc_stack, save a0 and a1 first, note s0 and sp are callee-saved
	mv s0, a0
	mv sp, a1

	call dealloc_stack # use asm's stack is safe

	mv a0, s0 # return value

	# restore callee-saved registers
	.set n, 8
	.rept 2
		LOAD_GP %n
		.set n, n+1
	.endr

	.set n, 18
	.rept 10
		LOAD_GP %n
		.set n, n+1
	.endr

	ld ra, 0*8(sp) # next pc
	ld sp, 2*8(sp) # old sp
	ret
