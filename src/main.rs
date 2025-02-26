mod matrix;
mod unumber;

use matrix::Matrix;
use std::time::Instant;
use unumber::UNumber;

fn unumber_example() {
    fn fib(n: &UNumber) -> UNumber {
        let mut a = unum!(0);
        let mut b = unum!(1);

        let mut i = unum!(0);
        while i < *n {
            let next = a + b.clone();

            a = b;
            b = next;
            i += unum!(1);
        }

        a
    }

    // UNumber Example:
    // When calculating numbers in the fibonacci sequence, the numbers can get very large.
    // Even using Rust's u128 type, (which most other languages don't have)
    // we can only calculate the 39th number in the sequence.
    // Becuase of this, when calucating fibonacci numbers, we need to use a type that can handle
    // arbitrarily large numbers. This is where UNumber comes in.
    println!("Starting!");
    let start = Instant::now();

    for i in 0..1_000 {
        fib(&unum!(i));
    }

    let duration = start.elapsed();
    println!("Finished!");
    println!(
        "Calculating the 1,000th fib number took {:?} seconds!",
        duration.as_secs_f64()
    );
}

fn matrix_example() {
    // Matrix Example:
    // Given system of equations:
    //  1.  x - 7y = -11
    //  2. 5x + 2y = -18

    // Matrix form: A * X = B
    //
    // Coefficient matrix A:
    // [ 1  -7 ]
    // [ 5   2 ]
    //
    // Variable vector X:
    // [ x ]
    // [ y ]
    //
    // Constant vector B:
    // [ -11 ]
    // [ -18 ]
    //
    // So, the system can be written as:
    //
    // [  1  -7 ] [ x ]   =   [ -11 ]
    // [  5   2 ] [ y ]       [ -18 ]

    // We can grab the solutions of x and y by taking A's inverse and multiplying it by B:
    // [ x ]   =   [  1  -7 ]^-1 [ -11 ]
    // [ y ]       [  5   2 ]    [ -18 ]
    let a = m2x2!(1.0, -7.0, 5.0, 2.0);
    let b = col_vec!(-11.0, -18.0);

    let a_inv = a.inverse().unwrap();
    let mut vars = a_inv * b;
    vars.fix_precision_errors();

    vars.display_vec();
    println!("Equations:");
    println!("x - 7y = -11");
    println!("5x + 2y = -18");
    println!("Solutions:");
    println!("x = {}", vars.get(0, 0).unwrap());
    println!("y = {}", vars.get(1, 0).unwrap());
}

fn main() {
    unumber_example();
    matrix_example();
}
