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
  ret i32 %2
}
