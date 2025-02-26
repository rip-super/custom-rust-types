use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt;
use std::ops::*;
use std::process::exit;

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
    pub fn new() -> Self {
        Self { digits: vec![] }
    }
}

impl From<&str> for UNumber {
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
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for UNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut leading_zero = true;
        for (i, digit) in self.digits.iter().enumerate() {
            if *digit != 0 {
                leading_zero = false;
            }
            if !leading_zero {
                if i > 0 && (self.digits.len() - i) % 3 == 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", digit)?;
            }
        }

        if leading_zero {
            write!(f, "0")?;
        }

        Ok(())
    }
}

impl PartialEq for UNumber {
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

impl AddAssign for UNumber {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl SubAssign for UNumber {
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl MulAssign for UNumber {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl DivAssign for UNumber {
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}
