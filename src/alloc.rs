use crate::up::UPSafeCell;
use core::ptr;
use lazy_static::lazy_static;
/// stack allocator
use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap, munmap, PROT_EXEC};
pub const STACK_SIZE: usize = 1024 * 16;
pub struct StackAllocator {
    stack_ptrs: Vec<u64>,
}

impl StackAllocator {
    pub fn new() -> Self {
        StackAllocator {
            stack_ptrs: Vec::new(),
        }
    }
    pub fn alloc(&mut self) -> u64 {
        unsafe {
            let addr = mmap(
                ptr::null_mut(),
                STACK_SIZE + 0x1000,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            println!("alloc: addr: {:#x}", addr as u64);
            if addr == libc::MAP_FAILED {
                panic!("Failed to allocate stack");
            }
            self.stack_ptrs.push(addr as u64);
            addr as u64
        }
    }
    pub fn dealloc(&mut self) {
        if let Some(addr) = self.stack_ptrs.pop() {
            unsafe {
                munmap(addr as *mut libc::c_void, STACK_SIZE);
            }
            println!("dealloc: addr: {:#x}\n", addr);
        } else {
            panic!("Nothing to deallocate!");
        }
    }
}

lazy_static! {
    pub static ref STACK_ALLOCATOR: UPSafeCell<StackAllocator> =
        unsafe { UPSafeCell::new(StackAllocator::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc_stack() -> u64 {
    let mut inner = STACK_ALLOCATOR.exclusive_access();
    inner.alloc()
}

#[unsafe(no_mangle)]
pub extern "C" fn dealloc_stack() {
    let mut inner = STACK_ALLOCATOR.exclusive_access();
    inner.dealloc();
}

pub fn alloc_mem(size: usize) -> u64 {
    unsafe {
        let addr = mmap(
            ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        );
        println!("alloc_mem: size: {}, addr: {:#x}", size, addr as u64);
        if addr == libc::MAP_FAILED {
            panic!("Failed to allocate stack");
        }
        addr as u64
    }
}
pub fn dealloc_mem(ptr: *mut u8, size: usize) {
    unsafe {
        munmap(ptr as *mut libc::c_void, size);
    }
}
