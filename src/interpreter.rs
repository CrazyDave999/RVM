use std::collections::HashMap;
use llvm_ir::{Function, Name, Operand};
use llvm_ir::Instruction;

pub struct InterpreterContext {
    pub virt_regs: HashMap<Name, isize>,
}
impl InterpreterContext {
    pub fn new() -> Self {
        InterpreterContext {
            virt_regs: HashMap::new(),
        }
    }
    pub fn get_operand(&self, op: &Operand) -> isize {
        match op {
            Operand::ConstantOperand(c) => {
                match &**c {
                    llvm_ir::Constant::Int {
                        value, ..
                    } => *value as isize,
                    _ => panic!("Unsupported constant type"),
                }
            },
            Operand::LocalOperand{name, ..} => {
                *self.virt_regs.get(name).unwrap()
            }
            _ => {
                panic!("Unsupported operand type");
            }
        }
    }
}

pub fn interpret_func(func: &Function) -> isize {
    func.parameters.iter().for_each(|param| {
        println!("Parameter: {:?}", param);
    });
    let mut ctx = InterpreterContext::new();
    for bb in func.basic_blocks.iter() {
        for inst in bb.instrs.iter() {
            interpret_inst(inst, &mut ctx);
        }
    }
    0
}

pub fn interpret_inst(inst: &Instruction, ctx: &mut InterpreterContext) {
    match inst {
        Instruction::Add(add) => {
            let lhs = ctx.get_operand(&add.operand0);
            let rhs = ctx.get_operand(&add.operand1);
            let res = lhs + rhs;
            ctx.virt_regs.insert(add.dest.clone(), res);
        }
        Instruction::Sub(sub) => {
            let lhs = ctx.get_operand(&sub.operand0);
            let rhs = ctx.get_operand(&sub.operand1);
            let res = lhs - rhs;
            ctx.virt_regs.insert(sub.dest.clone(), res);
        }
        Instruction::Mul(mul) => {
            let lhs = ctx.get_operand(&mul.operand0);
            let rhs = ctx.get_operand(&mul.operand1);
            let res = lhs * rhs;
            ctx.virt_regs.insert(mul.dest.clone(), res);
        }
        Instruction::SDiv(sdiv) => {
            let lhs = ctx.get_operand(&sdiv.operand0);
            let rhs = ctx.get_operand(&sdiv.operand1);
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs / rhs;
            ctx.virt_regs.insert(sdiv.dest.clone(), res);
        }
        Instruction::UDiv(udiv) => {
            let lhs = ctx.get_operand(&udiv.operand0) as usize;
            let rhs = ctx.get_operand(&udiv.operand1) as usize;
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs / rhs;
            ctx.virt_regs.insert(udiv.dest.clone(), res as isize);
        }
        Instruction::SRem(srem) => {
            let lhs = ctx.get_operand(&srem.operand0);
            let rhs = ctx.get_operand(&srem.operand1);
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs % rhs;
            ctx.virt_regs.insert(srem.dest.clone(), res);
        }
        Instruction::URem(urem) => {
            let lhs = ctx.get_operand(&urem.operand0) as usize;
            let rhs = ctx.get_operand(&urem.operand1) as usize;
            if rhs == 0 {
                panic!("Division by zero");
            }
            let res = lhs % rhs;
            ctx.virt_regs.insert(urem.dest.clone(), res as isize);
        }
        Instruction::And(and) => {
            let lhs = ctx.get_operand(&and.operand0);
            let rhs = ctx.get_operand(&and.operand1);
            let res = lhs & rhs;
            ctx.virt_regs.insert(and.dest.clone(), res);
        }
        Instruction::Or(or) => {
            let lhs = ctx.get_operand(&or.operand0);
            let rhs = ctx.get_operand(&or.operand1);
            let res = lhs | rhs;
            ctx.virt_regs.insert(or.dest.clone(), res);
        }
        Instruction::Xor(xor) => {
            let lhs = ctx.get_operand(&xor.operand0);
            let rhs = ctx.get_operand(&xor.operand1);
            let res = lhs ^ rhs;
            ctx.virt_regs.insert(xor.dest.clone(), res);
        }
        Instruction::Shl(shl) => {
            let lhs = ctx.get_operand(&shl.operand0);
            let rhs = ctx.get_operand(&shl.operand1);
            let res = lhs << rhs;
            ctx.virt_regs.insert(shl.dest.clone(), res);
        }
        Instruction::LShr(lshr) => {
            let lhs = ctx.get_operand(&lshr.operand0) as usize;
            let rhs = ctx.get_operand(&lshr.operand1) as usize;
            let res = lhs >> rhs;
            ctx.virt_regs.insert(lshr.dest.clone(), res as isize);
        }
        Instruction::AShr(ashr) => {
            let lhs = ctx.get_operand(&ashr.operand0);
            let rhs = ctx.get_operand(&ashr.operand1);
            let res = lhs >> rhs;
            ctx.virt_regs.insert(ashr.dest.clone(), res);
        }

        _ => {
            println!("Unsupported instruction: {:?}", inst);
        }
    }
}


