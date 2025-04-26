@.str1 = private unnamed_addr constant [13 x i8] c"HelloWorld!\0A\00", align 1
@.str2 = private unnamed_addr constant [13 x i8] c"hahahahahei\0A\00", align 1

define i32 @square(i32 %x) {
entry:
  %1 = mul i32 %x, %x
  ret i32 %1
}

define i32 @linear(i32 %a, i32 %x, i32 %b) {
entry:
  %1 = mul i32 %a, %x
  %2 = add i32 %1, %b
  ret i32 %2
}


define i32 @main() {
entry:
  %1 = call i32 @square(i32 5)
  %2 = call i32 @linear(i32 2, i32 %1, i32 3)
  %call = call signext i32 (ptr, ...) @print(ptr @.str1, ptr @.str2)
  call signext i32 @printInt(i32 %2)
  ret i32 %2
}

declare signext i32 @print(ptr noundef, ...)
declare signext i32 @printInt(i32)