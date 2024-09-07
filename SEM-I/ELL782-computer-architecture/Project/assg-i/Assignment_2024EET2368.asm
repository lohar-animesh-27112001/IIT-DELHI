.data
    A:    .float 2.0, 1.0, 3.0, 4.0, 5.0, 1.0, 0.0, 4.0, 3.0, 2.0, 3.0, 4.0, 0.0, 1.0 5.0, 4.0, 5.0, 1.0, 0.0, 3.0, 5.0, 2.0, 1.0, 4.0, 0.0
    I:    .float 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    B:    .space 100  # 5x5 matrix
    C:    .space 100  # 5x5 matrix
    D:    .space 100  # 5x5 matrix
    n:    .word 5     # Matrix size
    fmt:  .asciiz "%f "  # Format string for printing float with a space
no_solution_string:
    .asciiz "Inverse of the matrix does not exist.\n"

.text
.globl main
main:
    la t0, I
    la t1, B
    li t2, 0
    li t3, 25
    jal x1, copy_loop   # Copied I matrix into B
    la t0, A
    la t1, C
    li t2, 0
    li t3, 25
    jal x1, copy_loop   # Copied A matrix into C
    la t0, C
    la t1, B
    jal x1, row_swapping
    jal x1, gauss_elimination
    nop
.section:
copy_loop:
    beq t2, t3, end_copy   # If all elements are copied, exit loop

    flw ft0, 0(t0)      # Load the float element from I (use flw for floating-point load)
    fsw ft0, 0(t1)      # Store the float element into B (use fsw for floating-point store)

    addi t0, t0, 4      # Move to the next element in I
    addi t1, t1, 4      # Move to the next element in B
    addi t2, t2, 1      # Increment the index
    j copy_loop         # Jump back to the start of the loop

end_copy:
    # End of the program
    jalr x0, 0(ra)

row_swapping:
    li t2, 0  # Load the bit pattern for 0.0 (which is 0x00000000 for a 32-bit float) into an integer register t0
    flw f1, 0(t0)
    fcvt.w.s t3, f1
    beq t2, t3, non_zero_row
    jalr x0, 0(ra)

non_zero_row:
    li a2, 1
    lw a3, n #IMP
    li a4, 20
    loop:
        bge a2, a3, no_inverse  # if a2 = a3 = 5
        add t0, t0, a4
        flw f1, 0(t0)
        fcvt.w.s t3, f1
        bne t2, t3, swapping    # if t2 = 0, t3 != 0
        addi a2, a2, 1
        j loop  # jump to loop

swapping:
    la a2, C
    add a2, a2, a4     # Address of row what we got in matrix B
    lw a4, n #IMP      # Number of columns in the matrix IMP
    li a3, 0           # Column counter
    j swap_loop

swap_loop:
    beq a3, a4, end_swap # Exit loop after processing all columns

    # Load elements from both rows
    flw f0, 0(t0)    # Load element from row 1
    flw f1, 0(a2)    # Load element from row 3

    # Swap elements
    fsw f1, 0(a2)    # Store element from row 3 into row 1
    fsw f0, 0(t0)    # Store element from row 1 into row 3

    # Move to the next column
    addi t0, t0, 4    # Move to next element in row 1
    addi a2, a2, 4    # Move to next element in row 3
    addi a3, a3, 1    # Increment column counter

    j swap_loop

no_inverse:
    addi a0, x0, 4
    la a1, no_solution_string
    ecall

end_swap:
    jalr x0, 0(ra)

gauss_elimination:
    lw a2, n   # Load matrix size
    li a3, 1    # Load current row index
    li a4, 0    # Load current column index

# Converting upper triangular matrix
most_outer_loop:
    bge a4, a2, check_inverse
    outer_loop:
        bge a3, a2, end_outer_loop
        li a5, 1    # Load inner loop row
        add a5, a5, a4
        li a6, 4
        mul a6, a2, a6
        mul a6, a6, a3 # a6 = 20, 40, 60,...
        li a7, 0
        la t0, C
        la t1, B
        li t5, 4
        mul t5, t5, a2
        mul t5, t5, a4
        li t6, 4
        mul t6, t6, a4
        add t6, t5, t6
        add t0, t0, t6
        add t1, t1, t6

        flw f1, 0(t0)
        add t0, t0, a6
        flw f2, 0(t0)
        fdiv.s f1, f2, f1
        la t0, C
        la t1, B
        add t0, t0, t5
        add t1, t1, t5
        inner_loop:
            bge a5, a2, end_inner_loop
            # la t0, C
            # la t1, B
            la t2, C
            la t3, B
            add t2, t2, t5
            add t3, t3, t5
            add t2, t2, a6
            add t2, t2, a7
            add t3, t3, a6
            add t3, t3, a7

            add t0, t0, a7
            add t1, t1, a7
            flw f2, 0(t0)
            flw f3, 0(t1)
            flw f4, 0(t2)
            flw f5, 0(t3)

            fmul.s f6, f1, f2
            fmul.s f7, f1, f3

            fsub.s f6, f4, f6
            fsub.s f7, f5, f7

            fsw f6, 0(t2)
            fsw f7, 0(t3)

            addi a7, a7, 4
            addi a5, a5, 1
            j inner_loop
        
        end_inner_loop:
        addi a3, a3, 1
        j outer_loop
    end_outer_loop:
    addi a4, a4, 1
    j most_outer_loop

check_inverse_exist:
    la t1, B
    lw a2, n
    li a3, 1
    li a4, 4
    mul a5, a2, a4
    li a7, 0
    li t4, 1            # Load immediate value 1 into integer register t0
    fcvt.s.w f1, t4      # Move the value from t0 into floating-point register f1
        
    checking_loop:
        bge a3, a2, end_checking_loop
        mul t2, a5, a7
        mul t3, a4, a7
        add t3, t3, t2
        add t1, t1, t3
        flw f2, 0(t1)
        fmul.s f1, f1, f2
        addi a3, a3, 1
        addi a7, a7, 1
        j checking_loop

    end_checking_loop:
    li t4, 0            # Load immediate value 1 into integer register t0
    fcvt.s.w f2, t4
    feq.s t0, f1, f2
    beq t0, zero, no_inverse
    j converting_identity

# getting in form indentity matrix

converting_identity:
    # Load matrix size n
    la a0, n          # Load address of n
    lw a0, 0(a0)      # Load matrix size into a0

    # Load base addresses of matrices A and I
    la t0, C          # Base address of A (assuming C is the matrix A)
    la t1, B          # Base address of I (assuming B is the identity matrix)

    # Set up loop variables
    li t2, 0          # i = 0

outer_loop2:
    # Calculate the diagonal element offset
    mul t6, t2, a0    # t6 = i * n (row offset)
    add t6, t6, t2    # t6 = i * n + i (diagonal element index)
    slli t6, t6, 2    # t6 = (i * n + i) * 4 (byte offset)

    # Load the diagonal element A[i][i]
    add a2, t0, t6
    flw f0, 0(a2)

    # Check if diagonal element is not 1.0
    li a7, 1          # Load 1 into a7
    fcvt.s.w f1, a7   # Convert a7 to float in f1
    feq.s a6, f0, f1  # Compare f0 with 1.0
    bnez a6, skip_scaling  # If equal, skip scaling

    # Scaling the row to make diagonal element 1
    fdiv.s f0, f1, f0  # f0 = 1.0 / A[i][i]
    li a5, 0           # Column index j

inner_loop2:
    # Calculate the element offset A[i][j]
    mul a4, t2, a0    # a4 = i * n (row offset)
    add a4, a4, a5    # a4 = i * n + j (element index)
    slli a4, a4, 2    # a4 = (i * n + j) * 4 (byte offset)

    # Scale the element A[i][j]
    add a2, t0, a4
    flw f2, 0(a2)     # Load A[i][j] into f2
    fmul.s f2, f2, f0 # f2 = A[i][j] * (1.0 / A[i][i])
    
    add a2, t1, a4
    fsw f2, 0(a2)     # Store the result back to I[i][j]

    # Increment column index j
    addi a5, a5, 1    # j++
    blt a5, a0, inner_loop2 # Loop until j < n

skip_scaling:
    # Increment row index i
    addi t2, t2, 1    # i++
    blt t2, a0, outer_loop2 # Loop until i < n

    # Copy the result to matrix B (optional, based on your needs)
    li t2, 0          # Row index i
    li a5, 0          # Element index

copy_loop2:
    slli t6, a5, 2    # Byte offset
    add a2, t1, t6
    flw f2, 0(a2)     # Load element from I
    add a2, t0, t6
    fsw f2, 0(a2)     # Store element in A

    # Increment element index
    addi a5, a5, 1
    blt a5, a0, copy_loop2  # Loop until all elements are copied

    # Exit program
    li a0, 10         # Exit syscall

check_inverse:
    # Matrices are 5x5
    la t0, A # base address of matrix A
    la t1, B # base address of matrix B
    la t2, D # base address of matrix C
    # Matrix size: 5
    lw t3, n           # Load matrix size (n)

    # Initialize row index i
    li t4, 0           # i = 0

# Matrix multiplication:  D = C * B
outer_loop3:
    # Initialize column index j
    li t5, 0       # j = 0
inner_loop3:
    # Initialize sum for C[i][j]
    flw f0, 0(t2)  # Load C[i][j] (initially zero)
    li a4, 1          # Load 1 into a7
    fcvt.s.w f1, a4
    # Set f0 to 0.0 (sum initialization)

    # Initialize k index for dot product
    li t6, 0       # k = 0

dot_product:
    # Load A[i][k]
    mul a7, t4, t3    # a7 = i * n (row offset)
    add a7, a7, t6    # a7 = i * n + k (element index)
    slli a7, a7, 2    # a7 = (i * n + k) * 4 (byte offset)
    add a6, t0, a7    # a6 = base address of A + offset
    flw f1, 0(a6)     # Load A[i][k] into f1

    # Load B[k][j]
    mul a7, t6, t3    # a7 = k * n (row offset)
    add a7, a7, t5    # a7 = k * n + j (element index)
    slli a7, a7, 2    # a7 = (k * n + j) * 4 (byte offset)
    add a6, t1, a7    # a6 = base address of B + offset
    flw f2, 0(a6)     # Load B[k][j] into f2

    # Compute partial product and accumulate
    fmul.s f3, f1, f2 # f3 = A[i][k] * B[k][j]
    fadd.s f0, f0, f3 # f0 += A[i][k] * B[k][j]

    # Increment k index
    addi t6, t6, 1    # k++
    blt t6, t3, dot_product # Loop until k < n

    # Store result into C[i][j]
    mul a7, t4, t3    # a7 = i * n (row offset)
    add a7, a7, t5    # a7 = i * n + j (element index)
    slli a7, a7, 2    # a7 = (i * n + j) * 4 (byte offset)
    add a6, t2, a7    # a6 = base address of C + offset
    fsw f0, 0(a6)     # Store C[i][j] = sum

    # Increment column index j
    addi t5, t5, 1    # j++
    blt t5, t3, inner_loop3 # Loop until j < n

    # Increment row index i
    addi t4, t4, 1    # i++
    blt t4, t3, outer_loop3 # Loop until i < n

    # Exit program
    li a0, 10         # Exit syscall
printing_matrix:
    # Load the base address of the matrix
    la t0, B           # a0 = base address of matrix A
    la a1, fmt         # a1 = address of format string
    lw t1, n
    mul t2, t1, t1           # t0 = total number of elements (5x5)
    li a2, 0           # t2 = current index
    li a3, 4

print_matrix:
    beq a2, t2, end_print_matrix1
    flw f1, 0(t0)
    fcvt.w.s a5, f1
    li a0,1
    mv a1, a5
    ecall
    addi t0, t0, 4
    addi a2, a2, 1
    j print_matrix
end_print_matrix1:

printing_matrix2:
    # Load the base address of the matrix
    la t0, D           # a0 = base address of matrix A
    la a1, fmt         # a1 = address of format string
    lw t1, n
    mul t2, t1, t1           # t0 = total number of elements (5x5)
    li a2, 0           # t2 = current index
    li a3, 4

print_matrix2:
    beq a2, t2, end_print_matrix
    flw f1, 0(t0)
    fcvt.w.s a5, f1
    li a0,1
    mv a1, a5
    ecall
    addi t0, t0, 4
    addi a2, a2, 1
    j print_matrix2

end_print_matrix:
    ecall

