# now everything, except for sp, a0, is caller-saved, so no need to save s-registers in callee function

.altmacro
.macro SAVE_GP n
	sd x\n, \n*8(sp)
.endm
.macro LOAD_GP n
    ld x\n, \n*8(sp)
.endm

.extern FUNC_TABLE
.extern KERNEL_INFO
.extern trap_handler

	.global __call_fn
	.global __ret
	.align 2
# from asm to interpreter
__call_fn:
	# now t0 is the func index, t1 is old ra, ra is next pc of the call inst

	# save registers
	addi sp, sp, -8*32
	sd ra, 0*8(sp) # next pc
	sd t1, 1*8(sp) # old ra

	.set n, 8
	.rept 2
		SAVE_GP %n
		.set n, n+1
	.endr

	# no need to save a0, since it's for return value

	.set n, 11
	.rept 21
		SAVE_GP %n
		.set n, n+1
	.endr

	addi t2, sp, 8*32
	sd t2, 2*8(sp) # old sp

	# check if the func has been compiled
	slli t3, t0, 3 # offset
	ld t4, FUNC_TABLE(t3) # load the func address
	bnez t4, __call_compiled

	# if not compiled, let interpreter handle it
	# pub fn trap_handler(func_index: usize, ctx: *mut usize)
	mv a0, t0 # func index
	mv a1, sp # ctx

	# switch stack
	la t5, KERNEL_INFO
	ld sp, 0(t5)

	j trap_handler

__call_compiled:
	# if compiled, simply jump
	la ra, __ret
	jr t5

# from interpreter/asm to asm
__ret:
	# now sp points to the ctx

	# restore registers
	ld ra, 1*8(sp) # old ra

	.set n, 8
	.rept 2
		LOAD_GP %n
		.set n, n+1
	.endr

	# no need to restore a0, since it's for return value

	.set n, 11
	.rept 21
		LOAD_GP %n
		.set n, n+1
	.endr

	ld sp, 2*8(sp) # old sp

	ld t0, 0*8(sp) # next pc
	jr t0
