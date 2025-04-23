# RVM

The JIT virtual machine for LLVM IR, written in Rust.

Todo list:

- [x] Parse LLVM IR to basic blocks, etc.
- [ ] Interpretive execution 
- [ ] Compile certain BB
- [ ] Switch between interpretation and JIT
- [ ] Memory management
- [ ] JIT hello world
- [ ] Hotspot
- [ ] Optimization?

## Environment

The RVM will run on qemu with linux kernel, the architecture is riscv64.

## Mechanism

RVM code itself, runtime data, and JIT compiled machine code will all be placed in the memory.

Consider using mmap to allocate new memory space to place the newly compiled machine code.

### The steps of JIT compilation

1. Build asm for the target BB.
2. Allocate memory for the newly compiled machine code. Write the asm to the memory.
3. Store the context information of the current control flow (i.e. the RVM state).
4. Jump.

We use a table to indicate which BB has been compiled. RVM is responsible for maintaining this table.

### At the end of a compiled BB

 The last branch instruction should be modified to a small code fragment. Check the table to determine whether 
 jump directly to the target BB or return to RVM control flow.
 

## QEMU start command

```shell
qemu-system-riscv64 \
-machine virt \
-nographic \
-m 1G \
-kernel ~/linux-5.15.179/arch/riscv/boot/Image \
-drive file=~/dqib_riscv64-virt/image.qcow2,format=qcow2,if=none,id=hd0 \
-append "root=/dev/vda1 rw console=ttyS0" \
-netdev user,id=net0,hostfwd=tcp::2222-:22 -device virtio-net-device,netdev=net0 \
-device virtio-blk-device,drive=hd0
```