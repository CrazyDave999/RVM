use std::alloc::alloc;
use std::collections::HashMap;
use llvm_ir::{Function, Instruction, Terminator};
use std::sync::Arc;
use llvm_ir::types::Typed;
use crate::alloc::alloc_mem;

struct Reg(&'static str);
impl Reg {
    pub fn to_binary(&self) -> u32 {
        if self.0.starts_with('x') {
            self.0[1..].parse::<u32>().unwrap()
        } else {
            match self.0{
                "zero" => 0,
                "ra" => 1,
                "sp" => 2,
                "gp" => 3,
                "tp" => 4,
                "t0" => 5,
                "t1" => 6,
                "t2" => 7,
                "s0" => 8,
                "s1" => 9,
                "a0" => 10,
                "a1" => 11,
                "a2" => 12,
                "a3" => 13,
                "a4" => 14,
                "a5" => 15,
                "a6" => 16,
                "a7" => 17,
                "s2" => 18,
                "s3" => 19,
                "s4" => 20,
                "s5" => 21,
                "s6" => 22,
                "s7" => 23,
                "s8" => 24,
                "s9" => 25,
                "s10" => 26,
                "s11" => 27,
                "t3" => 28,
                "t4" => 29,
                "t5" => 30,
                "t6" => 31,
                _ => panic!("Unsupported register name"),
            }
        }
    }
}
enum InstType {
    R,
    I,
    IStar,
    L, // i.e. I, but different opcode
    S,
    B,
    U,
    J,
}
struct ASMInst {
    name: &'static str,
    rs1: Reg,
    rs2: Reg,
    rd: Reg,
    ty: InstType,
    imm: u32,
    label: Option<String>,
}

impl ASMInst {
    fn empty() -> Self {
        Self {
            name: "nop",
            rs1: Reg("x0"),
            rs2: Reg("x0"),
            rd: Reg("x0"),
            ty: InstType::R,
            imm: 0,
            label: None,
        }
    }
    fn from_inst(inst: &Instruction) -> Vec<Self> {
        vec![Self::empty()]
    }
    fn from_terminator(term: &Terminator) -> Vec<Self> {
        vec![Self::empty()]
    }
    fn to_binary(&self) -> u32 {
        let opcode = self.get_opcode();
        let (funct3, funct7) = self.get_funct();
        let rs1 = self.rs1.to_binary();
        let rs2 = self.rs2.to_binary();
        let rd = self.rd.to_binary();
        let shifted_imm = self.get_shifted_imm();
        match self.ty {
            InstType::R => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25)
            }
            InstType::I | InstType::L => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | shifted_imm
            }
            InstType::IStar => {
                opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | shifted_imm | (funct7 << 25)
            }
            InstType::S => opcode | shifted_imm | (funct3 << 12) | (rs1 << 15) | (rs2 << 20),
            InstType::B => opcode | shifted_imm | (funct3 << 12) | (rs1 << 15) | (rs2 << 20),
            InstType::U => opcode | (rd << 7) | shifted_imm,
            InstType::J => opcode | (rd << 7) | shifted_imm,
        }
    }
    fn to_string(&self) -> String {
        String::new()
    }
    fn get_funct(&self) -> (u32, u32) {
        match self.name {
            "add" => (0b000, 0b000_0000),
            "sub" => (0b000, 0b010_0000),
            "and" => (0b111, 0b000_0000),
            "or" => (0b110, 0b000_0000),
            "xor" => (0b100, 0b000_0000),
            "sll" => (0b001, 0b000_0000),
            "srl" => (0b101, 0b000_0000),
            "sra" => (0b101, 0b010_0000),
            "slt" => (0b010, 0b000_0000),
            "sltu" => (0b011, 0b000_0000),

            "addi" => (0b000, 0b000_0000),
            "andi" => (0b111, 0b000_0000),
            "ori" => (0b110, 0b000_0000),
            "xori" => (0b100, 0b000_0000),

            "slli" => (0b001, 0b000_0000),
            "srli" => (0b101, 0b000_0000),
            "srai" => (0b101, 0b010_0000),

            "slti" => (0b010, 0b000_0000),
            "sltiu" => (0b011, 0b000_0000),

            "ld" => (0b011, 0b000_0011),

            "sd" => (0b011, 0b010_0011),

            "beq" => (0b000, 0b000_0000),
            "bge" => (0b101, 0b000_0000),
            "bgeu" => (0b111, 0b000_0000),
            "blt" => (0b100, 0b000_0000),
            "bltu" => (0b110, 0b000_0000),
            "bne" => (0b001, 0b000_0000),

            "jal" => (0b000, 0b000_0000),
            "jalr" => (0b000, 0b000_0000),

            "auipc" => (0b000, 0b000_0000),
            "lui" => (0b000, 0b000_0000),

            _ => panic!("Unsupported instruction"),
        }
    }

    fn get_opcode(&self) -> u32 {
        match self.name {
            "jal" => 0b110_1111,
            "jalr" => 0b110_0111,
            "auipc" => 0b001_0111,
            "lui" => 0b011_0111,
            _ => match self.ty {
                InstType::R => 0b011_0011,
                InstType::I | InstType::IStar => 0b001_0011,
                InstType::L => 0b000_0011,
                InstType::S => 0b010_0011,
                InstType::B => 0b110_0011,
                _ => panic!("Unsupported instruction"),
            },
        }
    }
    fn get_shifted_imm(&self) -> u32 {
        match self.ty {
            InstType::I | InstType::L => (self.imm & ((1u32 << 12) - 1)) << 20,
            InstType::IStar => (self.imm & ((1u32 << 5) - 1)) << 20,
            InstType::S => {
                let imm = self.imm & ((1u32 << 12) - 1);
                ((imm & 0b1_1111) << 7) | ((imm >> 5) << 25)
            }
            InstType::B => {
                let imm = self.imm & ((1u32 << 13) - 1);
                ((imm & 0b1_0000_0000_0000) << 19)
                    | ((imm & 0b0_0111_1110_0000) << 20)
                    | ((imm & 0b0_0000_0001_1110) << 6)
                    | ((imm & 0b0_1000_0000_0000) >> 4)
            }
            InstType::U => {
                self.imm & 0xFFFF_F000
            }
            InstType::J => {
                let imm = self.imm & ((1u32 << 21) - 1);
                ((imm & 0b1_0000_0000_0000_0000_0000) << 11)
                    | ((imm & 0b0_0000_0000_0111_1111_1110) << 19)
                    | ((imm & 0b0_0000_0000_1000_0000_0000) << 11)
                    | (imm & 0b0_1111_1111_0000_0000_0000)
            }
            _ => 0,
        }
    }
    pub fn addi(rd: &'static str, rs: &'static str, imm: i32) -> Vec<Self>{
        // let mut res = Vec::new();
        // if imm >= -2048 && imm <= 2047 {
        //     res.push(Self {
        //         name: "addi",
        //         rs1: Reg(rs),
        //         rs2: Reg("x0"),
        //         rd: Reg(rd),
        //         ty: InstType::I,
        //         imm: imm as u32,
        //         label: None,
        //     });
        // } else {
        //     
        // }
        vec![]
    }
}

/// Build asm for a function， return the start addr of the func
/// functions that can compiled should not have phi
pub fn compile_func(func: Arc<Function>) -> u64 {
    let mut ptrs: HashMap<String, u64> = HashMap::new();
    let mut asm = Vec::new();
    // calculate stack size needed
    let mut stack_size = 0;
    let mut max_call_para_num = 0;
    for bb in func.basic_blocks.iter() {
        for inst in bb.instrs.iter() {
            match inst {
                Instruction::Add(_) => {}
                Instruction::Sub(_) => {}
                Instruction::Mul(_) => {}
                Instruction::UDiv(_) => {}
                Instruction::SDiv(_) => {}
                Instruction::URem(_) => {}
                Instruction::SRem(_) => {}
                Instruction::And(_) => {}
                Instruction::Or(_) => {}
                Instruction::Xor(_) => {}
                Instruction::Shl(_) => {}
                Instruction::LShr(_) => {}
                Instruction::AShr(_) => {}
                Instruction::FAdd(_) => {}
                Instruction::FSub(_) => {}
                Instruction::FMul(_) => {}
                Instruction::FDiv(_) => {}
                Instruction::FRem(_) => {}
                Instruction::FNeg(_) => {}
                Instruction::ExtractElement(_) => {}
                Instruction::InsertElement(_) => {}
                Instruction::ShuffleVector(_) => {}
                Instruction::ExtractValue(_) => {}
                Instruction::InsertValue(_) => {}
                Instruction::Alloca(_) => {}
                Instruction::Load(_) => {}
                Instruction::Store(_) => {}
                Instruction::Fence(_) => {}
                Instruction::CmpXchg(_) => {}
                Instruction::AtomicRMW(_) => {}
                Instruction::GetElementPtr(_) => {}
                Instruction::Trunc(_) => {}
                Instruction::ZExt(_) => {}
                Instruction::SExt(_) => {}
                Instruction::FPTrunc(_) => {}
                Instruction::FPExt(_) => {}
                Instruction::FPToUI(_) => {}
                Instruction::FPToSI(_) => {}
                Instruction::UIToFP(_) => {}
                Instruction::SIToFP(_) => {}
                Instruction::PtrToInt(_) => {}
                Instruction::IntToPtr(_) => {}
                Instruction::BitCast(_) => {}
                Instruction::AddrSpaceCast(_) => {}
                Instruction::ICmp(_) => {}
                Instruction::FCmp(_) => {}
                Instruction::Phi(_) => {}
                Instruction::Select(_) => {}
                Instruction::Freeze(_) => {}
                Instruction::Call(_) => {}
                Instruction::VAArg(_) => {}
                Instruction::LandingPad(_) => {}
                Instruction::CatchPad(_) => {}
                Instruction::CleanupPad(_) => {}
            }
        }
    }
    
    for bb in func.basic_blocks.iter() {
        for inst in bb.instrs.iter() {
            asm.extend(
                ASMInst::from_inst(inst)
            );
        }
        asm.extend(ASMInst::from_terminator(&bb.term));
    }
    let start_ptr = alloc_mem(asm.len() *4) ;
    let mut ptr =  start_ptr;
    for inst in asm.iter() {
        unsafe {
            *(ptr as *mut u32) = inst.to_binary();
        }
        ptr += 4;
    }
    start_ptr
}
