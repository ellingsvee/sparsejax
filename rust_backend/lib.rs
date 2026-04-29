use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

struct SelectedInverseCsc {
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

fn sort_pairs(rows: &mut Vec<usize>, vals: &mut Vec<f64>) {
    let mut pairs: Vec<(usize, f64)> = rows.iter().copied().zip(vals.iter().copied()).collect();
    pairs.sort_unstable_by_key(|&(row, _)| row);
    rows.clear();
    vals.clear();
    rows.reserve(pairs.len());
    vals.reserve(pairs.len());
    for (row, val) in pairs {
        rows.push(row);
        vals.push(val);
    }
}

fn compute_takahashi(
    indptr: ArrayView1<'_, i32>,
    indices: ArrayView1<'_, i32>,
    data: ArrayView1<'_, f64>,
    n: usize,
) -> SelectedInverseCsc {
    let diag: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|col| {
            let start = indptr[col] as usize;
            let end = indptr[col + 1] as usize;
            let mut d = 0.0;
            for idx in start..end {
                let row = indices[idx] as usize;
                if row == col {
                    d = data[idx];
                    break;
                }
            }
            d
        })
        .collect();

    let lower_pairs: Vec<(Vec<usize>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|col| {
            let start = indptr[col] as usize;
            let end = indptr[col + 1] as usize;
            let diag_col = diag[col];
            assert!(
                diag_col != 0.0,
                "Cholesky factor has a missing or zero diagonal"
            );

            let mut rows = Vec::new();
            let mut vals = Vec::new();
            for idx in start..end {
                let row = indices[idx] as usize;
                if row > col {
                    rows.push(row);
                    vals.push(data[idx] / diag_col);
                }
            }
            (rows, vals)
        })
        .collect();

    let (mut lower_rows, mut lower_vals): (Vec<Vec<usize>>, Vec<Vec<f64>>) =
        lower_pairs.into_iter().unzip();

    let mut upper_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for col in 0..n {
        for &row in &lower_rows[col] {
            upper_rows[row].push(col);
        }
    }

    lower_rows
        .par_iter_mut()
        .zip(lower_vals.par_iter_mut())
        .for_each(|(rows, vals)| sort_pairs(rows, vals));
    upper_rows
        .par_iter_mut()
        .for_each(|rows| rows.sort_unstable());

    let (l_indptr, l_indices, l_values) = flatten_columns(&lower_rows, &lower_vals);

    let z_cols: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|col| {
            let mut rows = Vec::with_capacity(upper_rows[col].len() + 1 + lower_rows[col].len());
            rows.extend_from_slice(&upper_rows[col]);
            rows.push(col);
            rows.extend_from_slice(&lower_rows[col]);
            rows
        })
        .collect();
    let (z_indptr, z_indices) = flatten_pattern(&z_cols);
    let mut z_values = vec![0.0; z_indices.len()];

    let z_diag_pos: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|col| {
            let start = z_indptr[col];
            let end = z_indptr[col + 1];
            let rel = z_indices[start..end]
                .binary_search(&col)
                .expect("selected inverse pattern must contain the diagonal");
            start + rel
        })
        .collect();
    for col in 0..n {
        z_values[z_diag_pos[col]] = 1.0 / (diag[col] * diag[col]);
    }

    let mut workspace = vec![0.0; n];
    let mut lmunch: Vec<usize> = (0..n)
        .map(|col| l_indptr[col + 1].saturating_sub(1))
        .collect();

    for j in (0..n).rev() {
        for p in z_diag_pos[j]..z_indptr[j + 1] {
            workspace[z_indices[p]] = z_values[p];
        }

        for p in (z_indptr[j]..z_diag_pos[j]).rev() {
            let k = z_indices[p];
            let mut zkj = 0.0;
            for up in l_indptr[k]..l_indptr[k + 1] {
                let i = l_indices[up];
                if i > k {
                    zkj -= l_values[up] * workspace[i];
                }
            }
            workspace[k] = zkj;
        }

        for p in (z_indptr[j]..z_diag_pos[j]).rev() {
            let k = z_indices[p];
            if l_indptr[k] == l_indptr[k + 1] || lmunch[k] < l_indptr[k] {
                continue;
            }
            if l_indices[lmunch[k]] != j {
                continue;
            }

            let ljk = l_values[lmunch[k]];
            if lmunch[k] > 0 {
                lmunch[k] -= 1;
            }

            for zp in z_diag_pos[k]..z_indptr[k + 1] {
                z_values[zp] -= workspace[z_indices[zp]] * ljk;
            }
        }

        for p in z_indptr[j]..z_indptr[j + 1] {
            let row = z_indices[p];
            z_values[p] = workspace[row];
            workspace[row] = 0.0;
        }
    }

    SelectedInverseCsc {
        indptr: z_indptr,
        indices: z_indices,
        values: z_values,
    }
}

fn flatten_columns(cols: &[Vec<usize>], vals: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = cols.len();
    let nnz: usize = cols.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    indptr.push(0);
    for col in 0..n {
        indices.extend_from_slice(&cols[col]);
        values.extend_from_slice(&vals[col]);
        indptr.push(indices.len());
    }
    (indptr, indices, values)
}

fn flatten_pattern(cols: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {
    let n = cols.len();
    let nnz: usize = cols.iter().map(Vec::len).sum();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(nnz);
    indptr.push(0);
    for rows in cols {
        indices.extend_from_slice(rows);
        indptr.push(indices.len());
    }
    (indptr, indices)
}

fn get_csc_value(z: &SelectedInverseCsc, row: usize, col: usize) -> f64 {
    let start = z.indptr[col];
    let end = z.indptr[col + 1];
    match z.indices[start..end].binary_search(&row) {
        Ok(pos) => z.values[start + pos],
        Err(_) => 0.0,
    }
}

#[pyfunction(name = "takahashi")]
fn takahashi_py<'py>(
    py: Python<'py>,
    l_indptr: PyReadonlyArray1<'py, i32>,
    l_indices: PyReadonlyArray1<'py, i32>,
    l_data: PyReadonlyArray1<'py, f64>,
    perm: PyReadonlyArray1<'py, i32>,
    n: usize,
) -> (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
) {
    // PyArrays to ndarray views
    let indptr = l_indptr.as_array();
    let indices = l_indices.as_array();
    let data = l_data.as_array();
    let p_vec = perm.as_array();

    let z = compute_takahashi(indptr, indices, data, n);

    // Flatten and Unpermute for return
    let selected_nnz = z.values.len();
    let mut col_for_pos = vec![0_usize; selected_nnz];
    for j in 0..n {
        for pos in z.indptr[j]..z.indptr[j + 1] {
            col_for_pos[pos] = j;
        }
    }

    let mut out_rows = vec![0_i32; selected_nnz];
    let mut out_cols = vec![0_i32; selected_nnz];
    let mut out_vals = vec![0.0_f64; selected_nnz];

    out_rows
        .par_iter_mut()
        .zip(out_cols.par_iter_mut())
        .zip(out_vals.par_iter_mut())
        .enumerate()
        .for_each(|(pos, ((row_out, col_out), val_out))| {
            *row_out = p_vec[z.indices[pos]];
            *col_out = p_vec[col_for_pos[pos]];
            *val_out = z.values[pos];
        });

    // Convert back to Python-managed Numpy arrays
    (
        out_rows.into_pyarray(py),
        out_cols.into_pyarray(py),
        out_vals.into_pyarray(py),
    )
}

#[pyfunction(name = "takahashi_masked")]
fn takahashi_masked_py<'py>(
    py: Python<'py>,
    l_indptr: PyReadonlyArray1<'py, i32>,
    l_indices: PyReadonlyArray1<'py, i32>,
    l_data: PyReadonlyArray1<'py, f64>,
    perm: PyReadonlyArray1<'py, i32>,
    query_rows: PyReadonlyArray1<'py, i32>,
    query_cols: PyReadonlyArray1<'py, i32>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let z = compute_takahashi(
        l_indptr.as_array(),
        l_indices.as_array(),
        l_data.as_array(),
        n,
    );
    let p_vec = perm.as_array();
    let rows = query_rows.as_array();
    let cols = query_cols.as_array();

    let mut inv_perm = vec![0_usize; n];
    for i in 0..n {
        inv_perm[p_vec[i] as usize] = i;
    }

    let nq = rows.len();
    let out_vals: Vec<f64> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let factor_row = inv_perm[rows[i] as usize];
            let factor_col = inv_perm[cols[i] as usize];
            get_csc_value(&z, factor_row, factor_col)
        })
        .collect();

    out_vals.into_pyarray(py)
}

#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(takahashi_py, m)?)?;
    m.add_function(wrap_pyfunction!(takahashi_masked_py, m)?)?;
    Ok(())
}
