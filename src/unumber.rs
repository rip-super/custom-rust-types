use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::ops::*;
use std::process::exit;

/// A macro to create a `UNumber` from a numeric literal.
///
/// This macro takes a numeric expression and converts it into a `UNumber` by first
/// converting it to a string and then parsing that string into a `UNumber`.
///
/// # Usage
///
/// The `unum!` macro can be used with any numeric literal or expression. For example:
///
/// ```
/// let num = unum!(123); // Creates a UNumber representing the number 123
/// let large_num = unum!(987654321); // Creates a UNumber representing 987654321
/// ```
///
/// # Panics
///
/// The macro will panic if the numeric literal cannot be converted into a valid
/// `UNumber` due to invalid formatting. Ensure that the input is a valid numeric value.
#[macro_export]
macro_rules! unum {
    ($num:expr) => {
        UNumber::from($num.to_string().as_str())
    };
}

#[derive(Debug, Clone)]
pub struct UNumber {
    digits: Vec<u8>,
}

impl UNumber {
    /// Creates a new `UNumber` instance.
    ///
    /// This constructor initializes a `UNumber` with an empty vector of digits.
    /// It is useful for creating a `UNumber` that can be populated later
    /// or for representing the number zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let num = UNumber::new();
    /// assert!(num.digits().is_empty()); // The digits vector should be empty
    /// ```
    pub fn new() -> Self {
        Self { digits: vec![] }
    }

    pub fn digits(&self) -> &Vec<u8> {
        &self.digits
    }
}

impl From<&str> for UNumber {
    /// Creates a `UNumber` from a string slice.
    ///
    /// This implementation of the `From` trait allows for the creation of a `UNumber`
    /// from a string representation of a number. The input string must contain only
    /// ASCII digits; any non-digit character will cause an error message to be printed,
    /// and the program will exit.
    ///
    /// # Panics
    ///
    /// If the input string contains non-digit characters, the function will print an
    /// error message and exit the program.
    ///
    /// # Examples
    ///
    /// ```
    /// let num = UNumber::from("12345");
    /// assert_eq!(num.digits(), vec![5, 4, 3, 2, 1]); // Digits are stored in reverse order
    ///
    /// // Invalid input example (will print error and exit):
    /// // let invalid_num = UNumber::from("123a45"); // Uncommenting this line will cause an error
    /// ```
    fn from(value: &str) -> Self {
        let mut digits = Vec::new();

        for c in value.chars() {
            if !c.is_ascii_digit() {
                eprintln!("Error: '{}' is not a digit", c);
                exit(1);
            }

            digits.push(c.to_digit(10).unwrap() as u8);
        }

        Self { digits }
    }
}

impl Default for UNumber {
    /// Creates a default `UNumber` instance.
    ///
    /// This implementation of the `Default` trait provides a way to create a `UNumber`
    /// initialized to its default state, which is an empty vector of digits.
    /// This is equivalent to calling `UNumber::new()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let num: UNumber = Default::default();
    /// assert!(num.digits().is_empty()); // The digits vector should be empty
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for UNumber {
    /// Formats the `UNumber` as a string.
    ///
    /// This implementation of the `Display` trait allows for easy conversion of a
    /// `UNumber` instance to a string representation. It outputs the number with
    /// commas as thousands separators and omits leading zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// let num = UNumber::from("123456789");
    /// assert_eq!(format!("{}", num), "123,456,789");
    ///
    /// let zero_num = UNumber::from("0000");
    /// assert_eq!(format!("{}", zero_num), "0");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut leading_zero = true;
        for (i, digit) in self.digits().iter().enumerate() {
            if *digit != 0 {
                leading_zero = false;
            }
            if !leading_zero {
                if i > 0 && (self.digits.len() - i) % 3 == 0 {
                    write!(f, ",")?; // Add comma as a thousands separator
                }
                write!(f, "{}", digit)?; // Write the digit
            }
        }

        if leading_zero {
            write!(f, "0")?; // Handle the case for zero
        }

        Ok(())
    }
}

impl PartialEq for UNumber {
    /// Checks for equality between two `UNumber` instances.
    ///
    /// This implementation allows for comparison of `UNumber` instances using the `==`
    /// operator. Two `UNumber` instances are considered equal if they have the same
    /// number of digits and corresponding digits are equal.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("123");
    /// let num2 = UNumber::from("123");
    /// let num3 = UNumber::from("456");
    /// assert!(num1 == num2); // num1 and num2 are equal
    /// assert!(num1 != num3); // num1 and num3 are not equal
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.digits.len() != other.digits.len() {
            return false;
        }

        for (a, b) in self.digits.iter().zip(other.digits.iter()) {
            if a != b {
                return false;
            }
        }

        true
    }
}

impl PartialOrd for UNumber {
    /// Compares two `UNumber` instances for ordering.
    ///
    /// This implementation allows for comparison of `UNumber` instances using comparison
    /// operators like `<`, `>`, and `<=`. The comparison is done first by the number of
    /// digits and then by the digits themselves.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("123");
    /// let num2 = UNumber::from("456");
    /// let num3 = UNumber::from("123");
    /// assert!(num1 < num2); // num1 is less than num2
    /// assert!(num1 == num3); // num1 is equal to num3
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.digits.len() < other.digits.len() {
            return Some(Ordering::Less);
        }

        if self.digits.len() > other.digits.len() {
            return Some(Ordering::Greater);
        }

        for (a, b) in self.digits.iter().zip(other.digits.iter()) {
            if a < b {
                return Some(Ordering::Less);
            }

            if a > b {
                return Some(Ordering::Greater);
            }
        }

        Some(Ordering::Equal)
    }
}

impl Add for UNumber {
    type Output = Self;

    /// Adds two `UNumber` instances.
    ///
    /// This implementation allows for the addition of two `UNumber` instances using the `+` operator.
    /// The addition is performed digit by digit, taking into account any carry from previous digits.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("12345");
    /// let num2 = UNumber::from("67890");
    /// let sum = num1 + num2;
    /// assert_eq!(format!("{}", sum), "80,235"); // Result should be formatted correctly
    /// ```
    fn add(self, other: Self) -> Self::Output {
        let mut result = UNumber::new();
        let mut carry = 0;

        for i in 0..self.digits.len().max(other.digits.len()) {
            let a = if i < self.digits.len() {
                self.digits[self.digits.len() - i - 1]
            } else {
                0
            };

            let b = if i < other.digits.len() {
                other.digits[other.digits.len() - i - 1]
            } else {
                0
            };

            let sum = a + b + carry;
            result.digits.insert(0, sum % 10);
            carry = sum / 10;
        }

        if carry > 0 {
            result.digits.insert(0, carry);
        }

        result
    }
}

impl Sub for UNumber {
    type Output = Self;

    /// Subtracts one `UNumber` from another.
    ///
    /// This implementation allows for the subtraction of two `UNumber` instances using the `-` operator.
    /// If the result would be negative, an error message is printed, and the program exits.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("100");
    /// let num2 = UNumber::from("45");
    /// let difference = num1 - num2;
    /// assert_eq!(format!("{}", difference), "55"); // Result should be formatted correctly
    /// ```
    fn sub(self, other: Self) -> Self::Output {
        if self < other {
            eprintln!("Error: Subtracting '{}' from '{}' would result in a negative value, but unsigned numbers cannot represent negative values.", self, other);
            exit(1);
        }

        let mut result = UNumber::new();
        let mut borrow = 0;

        for i in 0..self.digits.len().max(other.digits.len()) {
            let a = if i < self.digits.len() {
                self.digits[self.digits.len() - i - 1]
            } else {
                0
            };

            let b = if i < other.digits.len() {
                other.digits[other.digits.len() - i - 1]
            } else {
                0
            };

            let diff = if a >= b {
                a - b - borrow
            } else {
                a + 10 - b - borrow
            };

            result.digits.insert(0, diff);
            borrow = if a >= b { 0 } else { 1 };
        }

        result
    }
}

impl Mul for UNumber {
    type Output = Self;

    /// Multiplies two `UNumber` instances.
    ///
    /// This implementation allows for the multiplication of two `UNumber` instances using the `*` operator.
    /// The multiplication is performed using a grade school multiplication method, handling carry appropriately.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("123");
    /// let num2 = UNumber::from("456");
    /// let product = num1 * num2;
    /// assert_eq!(format!("{}", product), "56,088"); // Result should be formatted correctly
    /// ```
    fn mul(self, other: Self) -> Self::Output {
        let mut result = UNumber::new();

        for i in 0..self.digits.len() {
            let mut partial = UNumber::new();
            let mut carry = 0;

            for _ in 0..i {
                partial.digits.push(0);
            }

            for j in 0..other.digits.len() {
                let product = self.digits[self.digits.len() - i - 1]
                    * other.digits[other.digits.len() - j - 1]
                    + carry;
                partial.digits.insert(0, product % 10);
                carry = product / 10;
            }

            if carry > 0 {
                partial.digits.insert(0, carry);
            }

            result += partial;
        }

        result
    }
}

impl Div for UNumber {
    type Output = Self;

    /// Divides one `UNumber` by another.
    ///
    /// This implementation allows for the division of two `UNumber` instances using the `/` operator.
    /// If division by zero is attempted, an error message is printed, and the program exits.
    /// The quotient is returned as a new `UNumber` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("100");
    /// let num2 = UNumber::from("4");
    /// let quotient = num1 / num2;
    /// assert_eq!(format!("{}", quotient), "25"); // Result should be formatted correctly
    /// ```
    fn div(self, other: Self) -> Self::Output {
        if other == UNumber::from("0") {
            eprintln!("Error: Division by zero");
            exit(1);
        } else if self == other {
            return UNumber::from("1");
        } else if self < other {
            return UNumber::from("0");
        }

        let mut quotient = VecDeque::new();
        let mut remainder: u128 = 0;

        let divisor: u128 = other.digits.iter().fold(0, |acc, &d| acc * 10 + d as u128);

        for &digit in &self.digits {
            remainder = remainder * 10 + digit as u128;
            let q_digit = remainder / divisor;
            quotient.push_back(q_digit as u8);
            remainder -= q_digit * divisor;
        }

        while !quotient.is_empty() && quotient[0] == 0 {
            quotient.pop_front();
        }

        Self {
            digits: quotient.into(),
        }
    }
}

impl Rem for UNumber {
    type Output = Self;

    /// Computes the remainder of dividing one `UNumber` by another.
    ///
    /// This implementation allows using the `%` operator with `UNumber` instances.
    /// If division by zero is attempted, an error message is printed and the program exits.
    ///
    /// # Examples
    ///
    /// ```
    /// let num1 = UNumber::from("103");
    /// let num2 = UNumber::from("10");
    /// let rem = num1 % num2;
    /// assert_eq!(format!("{}", rem), "3"); // 103 % 10 = 3
    /// ```
    fn rem(self, rhs: Self) -> Self::Output {
        if rhs == UNumber::from("0") {
            eprintln!("Error: Division by zero");
            exit(1);
        } else if self < rhs {
            return self;
        } else if self == rhs {
            return UNumber::from("0");
        }

        let mut remainder: u128 = 0;
        let divisor: u128 = rhs.digits.iter().fold(0, |acc, &d| acc * 10 + d as u128);

        for &digit in &self.digits {
            remainder = remainder * 10 + digit as u128;
            let q_digit = remainder / divisor;
            remainder -= q_digit * divisor;
        }

        let mut digits = VecDeque::new();
        if remainder == 0 {
            digits.push_back(0);
        } else {
            let mut temp = remainder;
            let mut stack = Vec::new();
            while temp > 0 {
                stack.push((temp % 10) as u8);
                temp /= 10;
            }
            while let Some(d) = stack.pop() {
                digits.push_back(d);
            }
        }

        Self {
            digits: digits.into(),
        }
    }
}

impl AddAssign for UNumber {
    /// Adds another `UNumber` to the current instance in place.
    ///
    /// This implementation allows for the addition of another `UNumber` instance using the `+=` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut num = UNumber::from("10");
    /// num += UNumber::from("20");
    /// assert_eq!(format!("{}", num), "30"); // Result should be formatted correctly
    /// ```
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl SubAssign for UNumber {
    /// Subtracts another `UNumber` from the current instance in place.
    ///
    /// This implementation allows for the subtraction of another `UNumber` instance using the `-=` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut num = UNumber::from("30");
    /// num -= UNumber::from("10");
    /// assert_eq!(format!("{}", num), "20"); // Result should be formatted correctly
    /// ```
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl MulAssign for UNumber {
    /// Multiplies the current instance by another `UNumber` in place.
    ///
    /// This implementation allows for the multiplication of another `UNumber` instance using the `*=` operator.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut num = UNumber::from("3");
    /// num *= UNumber::from("4");
    /// assert_eq!(format!("{}", num), "12"); // Result should be formatted correctly
    /// ```
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl DivAssign for UNumber {
    /// Divides the current instance by another `UNumber` in place.
    ///
    /// This implementation allows for the division of another `UNumber` instance using the `/=` operator.
    /// Division by zero will print an error and exit the program.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut num = UNumber::from("100");
    /// num /= UNumber::from("5");
    /// assert_eq!(format!("{}", num), "20"); // Result should be formatted correctly
    /// ```
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

impl RemAssign for UNumber {
    /// Computes the remainder of `self` divided by `rhs`, storing the result in `self`.
    ///
    /// This allows using the `%=` operator on `UNumber` instances.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut num1 = UNumber::from("103");
    /// let num2 = UNumber::from("10");
    /// num1 %= num2;
    /// assert_eq!(format!("{}", num1), "3"); // 103 % 10 = 3
    /// ```
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}
