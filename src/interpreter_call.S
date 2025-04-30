	.globl __interpreter_call_asm
	.align 2
__interpreter_call_asm:
	# fn __interpreter_call_asm(fn_ptr: usize, para_num: usize, para_ptr: usize, new_sp: usize) -> isize
	mv t0, a0 # fn_ptr
	mv t1, a1 # para_num
	mv t2, a2 # para_ptr
	mv t3, a3 # new_sp

	# prepare para
	ld a0, 0*8(t2)
	ld a1, 1*8(t2)
	ld a2, 2*8(t2)
	ld a3, 3*8(t2)
	ld a4, 4*8(t2)
	ld a5, 5*8(t2)
	ld a6, 6*8(t2)
	ld a7, 7*8(t2)

	# save sp and ra
	addi sp, sp, -16
	sd s0, 0(sp)
	sd s1, 8(sp)
	mv s0, sp
	mv s1, ra

	# prepare space for para on stack
	addi t4, x0, 8
	blt t1, t4, __para_ok
	addi t4, t1, -8
	slli t4, t4, 3
	sub t3, t3, t4 # new_sp

	addi t4, x0, 8 # cur para
__para_loop:
	bgeu t4, t1, __para_ok
	slli t5, t4, 3 # offset of para_ptr
	add t5, t5, t2
	ld t5, 0(t5) # load cur para
	addi t6, t4, -8
	slli t6, t6, 3 # offset of new_sp
	add t6, t6, t3
	sd t5, 0(t6) # store cur para
	addi t4, t4, 1
	j __para_loop
__para_ok:
	# switch stack
	mv sp, t3

	jalr ra, 0(t0)

	# restore sp and ra
	mv sp, s0
	mv ra, s1
	ld s0, 0(sp)
	ld s1, 8(sp)
	addi sp, sp, 16

	ret