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
	# now t0 is the func index, ra is next pc of the call inst, paras are the same as C calling convention

	# check if the target function has been compiled
	slli t1, t0, 3 # offset
	la t2, FUNC_TABLE
	add t1, t1, t2
	ld t1, 0(t1) # load the func address
	bnez t1, __asm_call_asm

__asm_call_interpreter:
	# if not compiled, let the interpreter handle it
	addi sp, sp, -8*32
	sd ra, 1*8(sp) # next pc
	sd sp, 2*8(sp)

	.set n, 10
	.rept 8
		SAVE_GP %n
		.set n, n+1
	.endr

	# switch stack
	call alloc_stack
	li t2, 1024*16
	add t3, a0, t2 # new stack top

	# pub extern "C" fn asm_call_fn_handler(func_index: usize, ctx: *mut usize)
	ld a0, 5*8(sp) # func index
	mv a1, sp # ctx
	mv sp, t3 # new stack top

	j asm_call_fn_handler

__asm_call_asm:
	# if compiled, simply jump
	jr t1

__interpreter_ret_asm:
	# now a0 is the return value, a1 is the ctx

	# going to call dealloc_stack, save a0 and a1 first
	mv sp, a1 # ctx
	sd a0, 10*8(sp)

	call dealloc_stack # use asm's stack is safe

	ld a0, 10*8(sp) # return value
	ld ra, 1*8(sp) # next pc
	addi sp, sp, 8*32

	ret
