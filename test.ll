
define i32 @main() {
entry:
  %1 = add i32 1, 2
  %2 = add i32 3, 4
  %v1 = add i32 %1, %2
  ret i32 %v1
}
