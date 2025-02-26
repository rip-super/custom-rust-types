use num_traits::{Num, Float};
use std::fmt;
use std::ops::*;

#[macro_export]
/// Creates a new 2×2 matrix with the given values.
///
/// # Example
/// ```
/// let m = m2x2!(1, 2, 3, 4);
/// assert_eq!(m.get_data(), &vec![vec![1, 2], vec![3, 4]]);
/// ```
macro_rules! m2x2 {
    ($num1:expr, $num2:expr, $num3:expr, $num4:expr) => {
        Matrix::new(2, 2, vec![$num1, $num2, $num3, $num4])
    };
}

#[macro_export]
/// Creates a matrix from multiple row vectors.
///
/// # Example
/// ```
/// let m = mat!([1, 2], [3, 4]);
/// assert_eq!(m.get_data(), &vec![vec![1, 2], vec![3, 4]]);
/// ```
macro_rules! mat {
    ($([$($x:expr),*]),*) => {
        Matrix::from(vec![$(vec![$($x),*]),*])
    };
}

#[macro_export]
/// Applies a function to each element of a matrix and returns a new matrix.
///
/// # Example
/// ```
/// let m = mat!([1, 2], [3, 4]);
/// let squared = map_mat!(m, |x| x * x);
/// assert_eq!(squared.get_data(), &vec![vec![1, 4], vec![9, 16]]);
/// ```
macro_rules! map_mat {
    ($mat:expr, |$x:ident| $expr:expr) => {{
        let mut result = $mat.clone();
        for $x in result.iter_mut() {
            *$x = (|$x: &_| $expr)($x);
        }
        result
    }};
}

#[macro_export]
/// Creates a row vector (1×N matrix) from the given values.
///
/// # Example
/// ```
/// let v = row_vec!(1, 2, 3);
/// assert_eq!(v.get_data(), &vec![vec![1, 2, 3]]);
/// ```
macro_rules! row_vec {
    ($($x:expr),*) => {
        Matrix::from(vec![vec![$($x),*]])
    };
}

#[macro_export]
/// Creates a column vector (N×1 matrix) from the given values.
///
/// # Example
/// ```
/// let v = col_vec!(4, 5, 6);
/// assert_eq!(v.get_data(), &vec![vec![4], vec![5], vec![6]]);
/// ```
macro_rules! col_vec {
    ($($x:expr),*) => {
        Matrix::from(vec![$(vec![$x]),*])
    };
}

#[macro_export]
/// Creates a matrix of given dimensions and fills it with a specific value.
///
/// # Parameters
/// - `$rows`: Number of rows.
/// - `$cols`: Number of columns.
/// - `$val`: The value to fill the matrix with.
///
/// # Example
/// ```
/// let m = fill_mat!(2, 3, 7);
/// assert_eq!(m.get_data(), &vec![vec![7, 7, 7], vec![7, 7, 7]]);
/// ```
macro_rules! fill_mat {
    ($rows:expr, $cols:expr, $val:expr) => {
        Matrix::from(vec![vec![$val; $cols]; $rows])
    };
}

#[macro_export]
/// Creates a matrix from a flat list of values, arranged row-wise.
///
/// # Parameters
/// - `$rows`: Number of rows.
/// - `$cols`: Number of columns.
/// - `$($x:expr),*`: The list of values (must match `$rows * $cols`).
///
/// # Example
/// ```
/// let m = flat_mat!(2, 3, 1, 2, 3, 4, 5, 6);
/// assert_eq!(m.get_data(), &vec![vec![1, 2, 3], vec![4, 5, 6]]);
/// ```
macro_rules! flat_mat {
    ($rows:expr, $cols:expr, $($x:expr),*) => {{
        let data = vec![$($x),*];
        assert!(data.len() == $rows * $cols, "Invalid number of elements");
        let mut matrix = vec![vec![0; $cols]; $rows];
        for (i, val) in data.iter().enumerate() {
            matrix[i / $cols][i % $cols] = *val;
        }
        Matrix::from(matrix)
    }};
}

/// A generic 2D matrix structure supporting basic operations.
///
/// The `Matrix<T>` struct represents a mathematical matrix, stored as a nested `Vec<Vec<T>>`.
/// It provides various utility methods for accessing, modifying, and iterating over matrix elements.
///
/// # Type Parameters
/// - `T`: A numeric type that implements [`Num`](https://docs.rs/num/latest/num/trait.Num.html).
///
/// # Example
/// ```
/// use matrix::Matrix;
/// let m = Matrix::from(vec![vec![1, 2], vec![3, 4]]);
/// assert_eq!(m.get_data(), vec![vec![1, 2], vec![3, 4]]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T>
where
    T: Num,
{
    data: Vec<Vec<T>>,
}


impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Creates a new matrix with the given dimensions and values.
    ///
    /// # Parameters
    /// - `rows`: The number of rows.
    /// - `cols`: The number of columns.
    /// - `values`: A flat vector of elements, row-major order.
    ///
    /// # Returns
    /// A `Matrix<T>` with the specified dimensions and values 
    /// if the given parameters are valid, otherwise an empty matrix.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// assert_eq!(m.get(1, 0), Some(&3));
    /// ```
    pub fn new(rows: usize, cols: usize, values: Vec<T>) -> Self {
        if values.len() != rows * cols {
            return Self { data: vec![] };
        }
        let data = values.chunks(cols).map(|row| row.to_vec()).collect();
        Self { data }
    }

    /// Creates a matrix of zeros with the specified dimensions.
    ///
    /// # Parameters
    /// - `rows`: Number of rows.
    /// - `cols`: Number of columns.
    ///
    /// # Returns
    /// A `Matrix<T>` where all elements are `T::zero()`.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::zeros(2, 3);
    /// assert_eq!(m.get_data(), &vec![vec![0, 0, 0], vec![0, 0, 0]]);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec!(T::zero(); cols); rows],
        }
    }

    /// Creates a matrix of ones with the specified dimensions.
    ///
    /// # Parameters
    /// - `rows`: Number of rows.
    /// - `cols`: Number of columns.
    ///
    /// # Returns
    /// A `Matrix<T>` where all elements are `T::one()`.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::ones(2, 2);
    /// assert_eq!(m.get_data(), &vec![vec![1, 1], vec![1, 1]]);
    /// ```
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec!(T::one(); cols); rows],
        }
    }

    /// Creates an identity matrix of the given size (square matrix).
    ///
    /// # Parameters
    /// - `size`: The number of rows and columns.
    ///
    /// # Returns
    /// A square `Matrix<T>` where the diagonal elements are `T::one()` and all others are `T::zero()`.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::identity(3);
    /// assert_eq!(m.get_data(), &vec![
    ///     vec![1, 0, 0],
    ///     vec![0, 1, 0],
    ///     vec![0, 0, 1]
    /// ]);
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec!(T::zero(); size); size];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = T::one();
        }
        Self { data }
    }

    /// Returns the shape of the matrix as a tuple `(rows, cols)`.
    ///
    /// # Returns
    /// - A tuple `(usize, usize)` representing the number of rows and columns.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(3, 4, vec![1; 12]);
    /// assert_eq!(m.shape(), (3, 4));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.data.len(), self.data[0].len())
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Returns
    /// - `usize`: The number of rows.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::zeros(5, 3);
    /// assert_eq!(m.rows(), 5);
    /// ```
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Returns
    /// - `usize`: The number of columns.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::zeros(4, 6);
    /// assert_eq!(m.cols(), 6);
    /// ```
    pub fn cols(&self) -> usize {
        self.data[0].len()
    }

    /// Checks if the matrix is square (rows == columns).
    ///
    /// # Returns
    /// - `true` if the matrix is square, `false` otherwise.
    ///
    /// # Example
    /// ```
    /// let square_matrix = Matrix::identity(3);
    /// assert!(square_matrix.is_square());
    ///
    /// let non_square_matrix = Matrix::zeros(3, 4);
    /// assert!(!non_square_matrix.is_square());
    /// ```
    pub fn is_square(&self) -> bool {
        self.data.len() == self.data[0].len()
    }

    /// Checks if the matrix is empty.
    ///
    /// This function returns `true` if the matrix contains no rows and columns, 
    /// and `false` otherwise.
    ///
    /// # Returns
    /// - `true`: If the matrix is empty.
    /// - `false`: If the matrix contains one or more rows/columns.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(0, 0, vec![]); // Creating an empty matrix
    /// assert!(m.is_empty());
    ///
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// assert!(!m.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Retrieves a single element from the matrix at `(row, col)`, if valid.
    ///
    /// # Parameters
    /// - `row`: Row index (0-based).
    /// - `col`: Column index (0-based).
    ///
    /// # Returns
    /// - `Some(T)`: The element if the indices are valid.
    /// - `None`: If the indices are out of bounds.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// assert_eq!(m.get(0, 1), Some(2));
    /// assert_eq!(m.get(3, 0), None); // Out of bounds
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        self.data.get(row).and_then(|r| r.get(col)).copied()
    }

    /// Retrieves a full row from the matrix as a vector.
    ///
    /// # Parameters
    /// - `row`: The index of the row (0-based).
    ///
    /// # Returns
    /// - `Some(Vec<T>)`: A copy of the row if the index is valid.
    /// - `None`: If the index is out of bounds.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert_eq!(m.get_row(1), Some(vec![4, 5, 6]));
    /// assert_eq!(m.get_row(3), None); // Out of bounds
    /// ```
    pub fn get_row(&self, row: usize) -> Option<Vec<T>> {
        self.data.get(row).cloned()
    }

    /// Retrieves a full column from the matrix as a vector.
    ///
    /// # Parameters
    /// - `col`: The index of the column (0-based).
    ///
    /// # Returns
    /// - `Some(Vec<T>)`: A vector containing the column elements.
    /// - `None`: If the column index is out of bounds.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert_eq!(m.get_col(0), Some(vec![1, 4, 7]));
    /// assert_eq!(m.get_col(3), None); // Out of bounds
    /// ```
    pub fn get_col(&self, col: usize) -> Option<Vec<T>> {
        self.data.iter().map(|r| r.get(col).copied()).collect()
    }

    /// Returns a reference to the entire matrix data.
    ///
    /// # Returns
    /// - `&Vec<Vec<T>>`: The internal 2D representation of the matrix.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// assert_eq!(m.get_data(), &vec![vec![1, 2], vec![3, 4]]);
    /// ```
    pub fn get_data(&self) -> &Vec<Vec<T>> {
        &self.data
    }

    /// Sets the value of a specific element in the matrix.
    ///
    /// # Parameters
    /// - `row`: The row index (0-based).
    /// - `col`: The column index (0-based).
    /// - `value`: The new value to assign.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::zeros(2, 2);
    /// m.set(0, 1, 5);
    /// assert_eq!(m.get(0, 1), Some(5));
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if let Some(r) = self.data.get_mut(row) {
            if let Some(c) = r.get_mut(col) {
                *c = value;
            }
        }
    }

    /// Replaces an entire row in the matrix with new values.
    ///
    /// # Parameters
    /// - `row`: The row index (0-based).
    /// - `values`: A vector containing the new row values.
    ///
    /// # Returns
    /// - `Ok(())`: If the row was successfully replaced.
    /// - `Err(String)`: If `values.len()` does not match the matrix column count.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::zeros(3, 3);
    /// m.set_row(1, vec![4, 5, 6]).unwrap();
    /// assert_eq!(m.get_row(1), Some(vec![4, 5, 6]));
    /// ```
    pub fn set_row(&mut self, row: usize, values: Vec<T>) -> Result<(), String> {
        if row >= self.data.len() {
            return Err("Row index out of bounds.".to_string());
        }
        if let Some(r) = self.data.get_mut(row) {
            if r.len() != values.len() {
                return Err("New row must have the same number of columns.".to_string());
            }
            *r = values;
            Ok(())
        } else {
            Err("Row not found.".to_string()) // This case should not happen if the row index is valid.
        }
    }

    /// Replaces an entire column in the matrix with new values.
    ///
    /// # Parameters
    /// - `col`: The column index (0-based).
    /// - `values`: A vector containing the new column values.
    ///
    /// # Returns
    /// - `Ok(())`: If the column was successfully replaced.
    /// - `Err(String)`: If `values.len()` does not match the matrix row count or if the column index is out of bounds.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::zeros(3, 3);
    /// m.set_col(2, vec![7, 8, 9]).unwrap();
    /// assert_eq!(m.get_col(2), Some(vec![7, 8, 9]));
    /// ```
    pub fn set_col(&mut self, col: usize, values: Vec<T>) -> Result<(), String> {
        if col >= self.data[0].len() {
            return Err("Column index out of bounds.".to_string());
        }

        if values.len() != self.data.len() {
            return Err("New column must have the same number of rows.".to_string());
        }

        for (i, val) in values.iter().enumerate() {
            if let Some(r) = self.data.get_mut(i) {
                if let Some(c) = r.get_mut(col) {
                    *c = *val;
                }
            }
        }
        Ok(())
    }

    /// Multiplies every element in the matrix by a scalar value.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to multiply with.
    ///
    /// # Returns
    /// - A new `Matrix<T>` where each element is `original_value * scalar`.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let scaled = m.mul_scalar(2);
    /// assert_eq!(scaled.get_data(), &vec![vec![2, 4], vec![6, 8]]);
    /// ```
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let mut result = self.clone();
        for row in &mut result.data {
            for val in row.iter_mut() {
                *val = *val * scalar;
            }
        }
        result
    }

    /// Returns the transpose of the matrix.
    ///
    /// The transpose of a matrix is obtained by swapping rows with columns.
    ///
    /// # Returns
    /// - A new `Matrix<T>` with rows and columns swapped.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// let transposed = m.transpose();
    /// assert_eq!(transposed.get_data(), &vec![
    ///     vec![1, 4],
    ///     vec![2, 5],
    ///     vec![3, 6]
    /// ]);
    /// ```
    pub fn transpose(&self) -> Self {
        let rows = self.data.len();
        let cols = self.data[0].len();

        let mut transposed = Self::zeros(cols, rows);

        for i in 0..rows {
            for j in 0..cols {
                transposed.data[j][i] = self.data[i][j];
            }
        }

        transposed
    }

    /// Computes the inverse of the matrix if it exists.
    ///
    /// Uses **Gaussian elimination with partial pivoting** to compute the inverse.
    /// The matrix is augmented with the identity matrix, row operations are applied,
    /// and if successful, the right half of the augmented matrix is returned as the inverse.
    ///
    /// # Returns
    /// - `Some(Matrix<T>)`: If the matrix is invertible.
    /// - `None`: If the matrix is **not invertible** (singular) or **not square**.
    ///
    /// # Complexity
    /// - **Time:** `O(n³)`
    /// - **Space:** `O(n²)`
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![4, 7, 2, 6]);
    /// let inv = m.inverse().unwrap();
    /// let expected = Matrix::new(2, 2, vec![0.6, -0.7, -0.2, 0.4]);
    /// assert_eq!(inv.get_data(), expected.get_data());
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        if !self.is_square() {
            return None;
        }

        let n = self.data.len();
        let mut augmented = Self::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                augmented.set(i, j, self.data[i][j]);
            }
            augmented.set(i, n + i, T::one());
        }

        for i in 0..n {
            let mut pivot = augmented.get(i, i).unwrap();
            if pivot == T::zero() {
                for k in i + 1..n {
                    if augmented.get(k, i).unwrap() != T::zero() {
                        augmented.data.swap(i, k);
                        pivot = augmented.get(i, i).unwrap();
                        break;
                    }
                }
            }

            if pivot == T::zero() {
                return None;
            }

            for j in 0..2 * n {
                augmented.set(i, j, augmented.get(i, j).unwrap() / pivot);
            }

            for k in 0..n {
                if k == i {
                    continue;
                }

                let factor = augmented.get(k, i).unwrap();
                for j in 0..2 * n {
                    let val = augmented.get(k, j).unwrap() - factor * augmented.get(i, j).unwrap();
                    augmented.set(k, j, val);
                }
            }
        }

        let mut inverse = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inverse.set(i, j, augmented.get(i, n + j).unwrap());
            }
        }

        Some(inverse)
    }

    /// Attempts to correct floating-point arithmetic errors in the matrix by rounding each element.  
    ///
    /// Floating-point numbers can introduce small precision errors due to their binary representation,  
    /// leading to results such as `0.10000000000000009` instead of `0.1`.  
    /// This method **detects and rounds values dynamically** to mitigate these errors while  
    /// preserving as much precision as possible.
    ///
    /// # Algorithm  
    /// - Iterates over each element in the matrix.
    /// - For each element, determines the optimal decimal precision for rounding.
    /// - If excessive repeating digits are detected, the value is rounded to a stable representation.  
    /// - Otherwise, full precision is maintained.  
    ///
    /// # Returns  
    /// - Modifies the matrix **in place**, ensuring elements have corrected precision.  
    ///
    /// # Time Complexity  
    /// - `O(n)` (linear in the number of elements)
    ///
    /// # Example  
    /// ```
    /// let mut m = Matrix::new(2, 2, vec![0.10000000000000009, 3.141592653589793, 2.000000000000004, -0.49999999999999994]);
    /// m.fix_precision_errors();
    /// let expected = Matrix::new(2, 2, vec![0.1, 3.141592653589793, 2.0, -0.5]);
    /// assert_eq!(m.get_data(), expected.get_data());
    /// ```
    ///
    /// # Constraints  
    /// - Only applies to floating-point types (`f32`, `f64`).
    /// - Does **not** modify integer values.
    pub fn fix_precision_errors(&mut self) where T: Float {
        self.iter_mut().for_each(|x| {
            let precision = Self::detect_floating_point_error(*x);
            let factor = T::from(10.0_f64.powi(precision)).unwrap();
            *x = (*x * factor).round() / factor;
        });
    }

    fn detect_floating_point_error(num: T) -> i32 where T: Float{
        let num_str = format!("{:.15}", num.to_f64().unwrap());
        let decimal_part = num_str.split('.').nth(1).unwrap_or("");

        let mut repeating_count = 0;
        let mut last_char = None;

        for c in decimal_part.chars() {
            if Some(c) == last_char {
                repeating_count += 1;
            } else {
                repeating_count = 0;
            }
            last_char = Some(c);

            if repeating_count >= 5 {
                return 5;
            }
        }

        15
    }

    /// Computes the determinant of the matrix.
    ///
    /// Uses **cofactor expansion** for small matrices and a recursive approach for larger ones.
    ///
    /// # Returns
    /// - `Some(T)`: The determinant if the matrix is square.
    /// - `None`: If the matrix is not square.
    ///
    /// # Complexity
    /// - Base cases (`1x1` and `2x2` matrices): `O(1)`
    /// - General case (recursive cofactor expansion): `O(n!)`
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(3, 3, vec![
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9
    /// ]);
    /// assert_eq!(m.determinant(), Some(0)); // Singular matrix
    /// ```
    #[rustfmt::skip]
    pub fn determinant(&self) -> Option<T> 
    where
        T: Neg<Output = T>, 
    {
        if !self.is_square() {
            return None;
        }

        let n = self.data.len();

        match n {
            0 => None,
            1 => Some(self.data[0][0]),
            2 => Some(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]),
            _ => Some(self.det(n)),
        }
    }

    fn det(&self, n: usize) -> T
    where
        T: Neg<Output = T>,
    {
        // cofactor expansion
        // O(n!) time complexity
        // TODO: try to improve this
        let mut det = T::zero();
        for col in 0..n {
            let mut sub_matrix = Self::zeros(n - 1, n - 1);
            for i in 1..n {
                let mut subcol = 0;
                for j in 0..n {
                    if j == col {
                        continue;
                    }

                    sub_matrix.set(i - 1, subcol, self.data[i][j]);
                    subcol += 1;
                }
            }

            let sign = if col % 2 == 0 { T::one() } else { -T::one() };
            det = det + sign * self.data[0][col] * sub_matrix.determinant().unwrap();
        }

        det
    }

    /// Raises the matrix to the power of `exp`.
    ///
    /// This function performs matrix exponentiation. It can only be applied to square matrices.
    ///
    /// # Parameters
    /// - `exp`: The exponent to raise the matrix to. Should be a value of type `T` that implements `PartialOrd`.
    ///
    /// # Returns
    /// - `Some(Self)`: A new matrix resulting from raising the original matrix to the power of `exp`.
    /// - `None`: If the matrix is not square.
    ///
    /// # Examples
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let result = m.pow(2);
    /// assert_eq!(result, Some(Matrix::new(2, 2, vec![7, 10, 15, 22])));
    ///
    /// let m = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    /// let result = m.pow(2);
    /// assert_eq!(result, None); // Not a square matrix
    /// ```
    pub fn pow(&self, exp: T) -> Option<Self>
    where
        T: std::cmp::PartialOrd,
    {
        if !self.is_square() {
            return None;
        } else if exp == T::zero() {
            return Some(Self::identity(self.data.len()));
        } else if exp == T::one() {
            return Some(self.clone());
        }

        let mut result = self.clone();
        let mut i = exp - T::one();
        while i > T::zero() {
            result = result * self.clone();
            i = i - T::one();
        }

        Some(result)
    }

    /// Displays the matrix in a human-readable format.
    ///
    /// This function prints each element of the matrix to the console, 
    /// with elements of each row on a new line.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// m.display(); // Output:
    /// // 1 2 
    /// // 3 4 
    /// ```
    pub fn display(&self) where T: fmt::Display {
        for row in &self.data {
            for val in row {
                print!("{} ", val);
            }
            println!();
        }
    }

    /// Displays the matrix as a vector of vectors.
    ///
    /// This function prints the matrix in a vector format, 
    /// showing each row as a separate vector.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// m.display_vec(); // Output:
    /// // [
    /// //   [1, 2],
    /// //   [3, 4]
    /// // ]
    /// ```
    pub fn display_vec(&self) where T: fmt::Display {
        println!("[");
        for (row_index, row) in self.data.iter().enumerate() {
            print!("  [");
            for (i, val) in row.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}", val);
            }
            print!("]");
            if row_index < self.data.len() - 1 {
                println!(",");
            } else {
                println!();
            }
        }
        println!("]")
    }
}

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Returns an iterator over the elements of the matrix.
    ///
    /// This iterator allows you to access each element in the matrix in a 
    /// row-major order.
    ///
    /// # Returns
    /// An iterator yielding references to the elements of the matrix.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let elements: Vec<_> = m.iter().collect();
    /// assert_eq!(elements, vec![&1, &2, &3, &4]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flat_map(|row| row.iter())
    }

    /// Returns a mutable iterator over the elements of the matrix.
    ///
    /// This mutable iterator allows you to access and modify each element in the 
    /// matrix in a row-major order.
    ///
    /// # Returns
    /// A mutable iterator yielding mutable references to the elements of the matrix.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// for val in m.iter_mut() {
    ///     *val += 1; // Increment each element by 1
    /// }
    /// let elements: Vec<_> = m.iter().collect();
    /// assert_eq!(elements, vec![&2, &3, &4, &5]);
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().flat_map(|row| row.iter_mut())
    }

    /// Returns an iterator over the rows of the matrix.
    ///
    /// This iterator yields references to each row (as a `Vec<T>`) in the matrix.
    ///
    /// # Returns
    /// An iterator yielding references to the rows of the matrix.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let rows: Vec<_> = m.iter_rows().collect();
    /// assert_eq!(rows, vec![&vec![1, 2], &vec![3, 4]]);
    /// ```
    pub fn iter_rows(&self) -> impl Iterator<Item = &Vec<T>> {
        self.data.iter()
    }

    /// Returns a mutable iterator over the rows of the matrix.
    ///
    /// This mutable iterator allows you to access and modify each row in the 
    /// matrix.
    ///
    /// # Returns
    /// A mutable iterator yielding mutable references to the rows of the matrix.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// for row in m.iter_rows_mut() {
    ///     row.push(5); // Add a new element to each row
    /// }
    /// assert_eq!(m.data, vec![vec![1, 2, 5], vec![3, 4, 5]]);
    /// ```
    pub fn iter_rows_mut(&mut self) -> impl Iterator<Item = &mut Vec<T>> {
        self.data.iter_mut()
    }

    /// Returns an iterator over the columns of the matrix.
    ///
    /// This iterator yields vectors containing the elements of each column 
    /// in the matrix.
    ///
    /// # Returns
    /// An iterator yielding vectors of the elements in each column.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let cols: Vec<_> = m.iter_cols().collect();
    /// assert_eq!(cols, vec![vec![1, 3], vec![2, 4]]);
    /// ```
    pub fn iter_cols(&self) -> impl Iterator<Item = Vec<T>> + '_ {
        (0..self.data[0].len()).map(move |col| self.get_col(col).unwrap())
    }

    /// Returns a mutable iterator over the columns of the matrix.
    ///
    /// This mutable iterator allows you to access and modify each column in the 
    /// matrix.
    ///
    /// # Returns
    /// A mutable iterator yielding vectors of the elements in each column.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// for mut col in m.iter_cols_mut() {
    ///     col.push(5); // Add a new element to each column
    /// }
    /// assert_eq!(m.data, vec![vec![1, 2, 5], vec![3, 4, 5]]);
    /// ```
    pub fn iter_cols_mut(&mut self) -> impl Iterator<Item = Vec<T>> + '_ {
        (0..self.data[0].len()).map(move |col| self.get_col(col).unwrap())
    }
}

impl<T> IntoIterator for Matrix<T>
where
    T: Num,
{
    /// Creates an iterator that consumes the matrix and yields its elements.
    ///
    /// This implementation allows you to iterate over the elements of the matrix
    /// in a row-major order. The matrix will be consumed in the process.
    ///
    /// # Returns
    /// An iterator yielding the elements of the matrix.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// let elements: Vec<_> = m.into_iter().collect();
    /// assert_eq!(elements, vec![1, 2, 3, 4]);
    /// ```
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().flat_map(|row| row.into_iter()).collect::<Vec<T>>().into_iter()
    }
}

impl<T> IntoIterator for &mut Matrix<T>
where
    T: Num + Copy,
{
    /// Creates a mutable iterator that allows modification of the matrix elements.
    ///
    /// This implementation enables you to iterate over and modify the elements
    /// of the matrix in a row-major order without consuming the matrix.
    ///
    /// # Returns
    /// A mutable iterator yielding mutable references to the elements of the matrix.
    ///
    /// # Example
    /// ```
    /// let mut m = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// for val in m.iter_mut() {
    ///     *val *= 2; // Double each element
    /// }
    /// let elements: Vec<_> = m.into_iter().collect();
    /// assert_eq!(elements, vec![2, 4, 6, 8]);
    /// ```
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut().flat_map(|row| row.iter_mut().map(|val| *val)).collect::<Vec<T>>().into_iter()
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Num,
{
    /// Creates a new `Matrix<T>` from a vector of vectors.
    ///
    /// This implementation allows you to convert a `Vec<Vec<T>>` into a `Matrix<T>`.
    ///
    /// # Parameters
    /// - `data`: A 2D vector containing the elements of the matrix.
    ///
    /// # Example
    /// ```
    /// let data = vec![vec![1, 2], vec![3, 4]];
    /// let matrix: Matrix<_> = Matrix::from(data);
    /// assert_eq!(matrix.get(0, 1), Some(&2));
    /// ```
    fn from(data: Vec<Vec<T>>) -> Self {
        Self { data }
    }
}

impl<T> Default for Matrix<T>
where
    T: Num + Copy,
{
    /// Creates a default `Matrix<T>` with zero rows and columns.
    ///
    /// This implementation allows you to create an empty matrix using the default 
    /// constructor.
    ///
    /// # Example
    /// ```
    /// let matrix: Matrix<i32> = Matrix::default();
    /// assert!(matrix.is_empty());
    /// ```
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl<T> fmt::Display for Matrix<T>
where
    T: Num + Copy + fmt::Display,
{
    /// Formats the matrix for display.
    ///
    /// This implementation allows you to print the matrix in a human-readable 
    /// format, displaying each element in a row-major order.
    ///
    /// # Example
    /// ```
    /// let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    /// println!("{}", matrix);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in &self.data {
            for val in row {
                write!(f, "{} ", val)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl<T> Add for Matrix<T>
where
    T: Num + Copy,
{
    /// Adds two matrices element-wise.
    ///
    /// This implementation allows you to add two matrices of the same dimensions.
    ///
    /// # Parameters
    /// - `other`: The matrix to add to `self`.
    ///
    /// # Returns
    /// A new `Matrix<T>` that is the element-wise sum of `self` and `other`.
    ///
    /// # Panics
    /// If the dimensions of the two matrices do not match.
    ///
    /// # Example
    /// ```
    /// let a = m2x2!(1, 2, 3, 4);
    /// let b = m2x2!(5, 6, 7, 8);
    /// let sum = a + b;
    /// assert_eq!(sum.get_data(), &vec![vec![6, 8], vec![10, 12]]);
    /// ```
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let rows = self.data.len();
        let cols = self.data[0].len();

        if rows != other.data.len() || cols != other.data[0].len() {
            panic!("Matrix dimensions do not match.");
        }

        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let val1 = self.data[i][j];
                let val2 = other.data[i][j];
                result.data[i][j] = val1 + val2;
            }
        }
        result
    }
}

impl<T> Sub for Matrix<T>
where
    T: Num + Copy,
{
    /// Subtracts one matrix from another element-wise.
    ///
    /// This implementation allows you to subtract the matrix `other` from `self`.
    ///
    /// # Parameters
    /// - `other`: The matrix to subtract from `self`.
    ///
    /// # Returns
    /// A new `Matrix<T>` that is the element-wise difference of `self` and `other`.
    ///
    /// # Panics
    /// If the dimensions of the two matrices do not match.
    ///
    /// # Example
    /// ```
    /// let a = m2x2!(5, 6, 7, 8);
    /// let b = m2x2!(1, 2, 3, 4);
    /// let difference = a - b;
    /// assert_eq!(difference.get_data(), &vec![vec![4, 4], vec![4, 4]]);
    /// ```
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let rows = self.data.len();
        let cols = self.data[0].len();

        if rows != other.data.len() || cols != other.data[0].len() {
            panic!("Matrix dimensions do not match.");
        }

        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let val1 = self.data[i][j];
                let val2 = other.data[i][j];
                result.data[i][j] = val1 - val2;
            }
        }
        result
    }
}

impl<T> Mul for Matrix<T>
where
    T: Num + Copy,
{
    /// Multiplies two matrices using the matrix multiplication rule.
    ///
    /// This implementation allows you to multiply two matrices where the number of columns
    /// in the first matrix matches the number of rows in the second matrix.
    ///
    /// # Parameters
    /// - `other`: The matrix to multiply with `self`.
    ///
    /// # Returns
    /// A new `Matrix<T>` that is the product of `self` and `other`.
    ///
    /// # Panics
    /// If the inner dimensions do not match for matrix multiplication.
    ///
    /// # Example
    /// ```
    /// let a = m2x2!(1, 2, 3, 4);
    /// let b = m2x2!(5, 6, 7, 8);
    /// let product = a * b;
    /// assert_eq!(product.get_data(), &vec![vec![19, 22], vec![43, 50]]);
    /// ```
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let rows = self.data.len();
        let cols = other.data[0].len();
        let inner_dim = self.data[0].len();

        if inner_dim != other.data.len() {
            panic!("Matrix dimensions do not match.");
        }

        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::zero();
                for k in 0..inner_dim {
                    sum = sum + (self.data[i][k] * other.data[k][j]);
                }
                result.data[i][j] = sum;
            }
        }
        result
    }
}