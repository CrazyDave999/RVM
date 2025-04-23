use crate::up::UPSafeCell;
use core::ptr;
use lazy_static::lazy_static;
/// stack allocator
use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap, munmap};
pub const STACK_SIZE: usize = 1024 * 16;
pub struct StackAllocator {
    stack_ptrs: Vec<usize>,
}

impl StackAllocator {
    pub fn new() -> Self {
        StackAllocator {
            stack_ptrs: Vec::new(),
        }
    }
    pub fn alloc(&mut self) -> usize {
        unsafe {
            let addr = mmap(
                ptr::null_mut(),
                STACK_SIZE,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            if addr == libc::MAP_FAILED {
                panic!("Failed to allocate stack");
            }
            self.stack_ptrs.push(addr as usize);
            addr as usize
        }
    }
    pub fn dealloc(&mut self) {
        if let Some(addr) = self.stack_ptrs.pop() {
            unsafe {
                munmap(addr as *mut libc::c_void, STACK_SIZE);
            }
        }
    }
}

lazy_static! {
    pub static ref STACK_ALLOCATOR: UPSafeCell<StackAllocator> =
        unsafe { UPSafeCell::new(StackAllocator::new()) };
}

#[no_mangle]
pub extern "C" fn alloc_stack() -> usize {
    let mut inner = STACK_ALLOCATOR.exclusive_access();
    inner.alloc()
}

#[no_mangle]
pub extern "C" fn dealloc_stack() {
    let mut inner = STACK_ALLOCATOR.exclusive_access();
    inner.dealloc();
}
