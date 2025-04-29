@.str1 = private unnamed_addr constant [13 x i8] c"HelloWorld!\0A\00", align 1
@.str2 = private unnamed_addr constant [2 x i8] c" \00", align 1

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
    %x = alloca [10 x i32], align 4
    %p0 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i32 0, i32 0
    store i32 0, i32* %p0, align 4
    %p1 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i32 0, i32 1
    store i32 1, i32* %p1, align 4
    %p2 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i32 0, i32 2
    store i32 2, i32* %p2, align 4
    %p3 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i32 0, i32 3
    store i32 3, i32* %p3, align 4
    %p4 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i32 0, i32 4
    store i32 4, i32* %p4, align 4

    %v0 = load i32, i32* %p0, align 4
    call i32 @printInt(i32 %v0)
    call i32 @print(ptr @.str2)

    %v1 = load i32, i32* %p1, align 4
    call i32 @printInt(i32 %v1)
    call i32 @print(ptr @.str2)

    %v2 = load i32, i32* %p2, align 4
    call i32 @printInt(i32 %v2)
    call i32 @print(ptr @.str2)

    %v3 = load i32, i32* %p3, align 4
    call i32 @printInt(i32 %v3)
    call i32 @print(ptr @.str2)

    %v4 = load i32, i32* %p4, align 4
    call i32 @printInt(i32 %v4)
    call i32 @print(ptr @.str2)

    call i32 @print(ptr @.str1)
    ret i32 66
}

define i32 @print(ptr noundef %str) {
entry:
    %v0 = call i32 @..print(ptr noundef %str)
    ret i32 %v0
}

define i32 @printInt(i32 %x) {
entry:
    %v0 = call i32 @..printInt(i32 %x)
    ret i32 %v0
}

declare signext i32 @..print(ptr noundef)
declare signext i32 @..printInt(i32)