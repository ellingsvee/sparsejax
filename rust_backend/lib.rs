use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn get_selected(rows: &[Vec<usize>], vals: &[Vec<f64>], a: usize, b: usize) -> f64 {
    let (row, col) = if a <= b { (a, b) } else { (b, a) };
    match rows[col].binary_search(&row) {
        Ok(pos) => vals[col][pos],
        Err(_) => 0.0,
    }
}

fn set_selected(rows: &[Vec<usize>], vals: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    if let Ok(pos) = rows[col].binary_search(&row) {
        vals[col][pos] = value;
    }
}

fn compute_takahashi(
    indptr: ArrayView1<'_, i32>,
    indices: ArrayView1<'_, i32>,
    data: ArrayView1<'_, f64>,
    n: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    // Precompute structure
    let mut col_diag = vec![0.0; n];
    let mut col_sub_indices: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut col_sub_values: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;

        for idx in start..end {
            let row = indices[idx] as usize;
            let val = data[idx];

            if row == i {
                col_diag[i] = val;
            } else if row > i {
                col_sub_indices[i].push(row);
                col_sub_values[i].push(val);
                row_to_cols[row].push(i);
            }
        }
    }

    // Takahashi algorithm using the selected inverse's upper-triangular CSC
    // pattern. Column j contains row j plus every i where L[j, i] is nonzero.
    let mut z_rows: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut z_vals: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut rows = row_to_cols[j].clone();
        rows.push(j);
        z_vals.push(vec![0.0; rows.len()]);
        z_rows.push(rows);
    }

    for j in (0..n).rev() {
        let l_jj = col_diag[j];
        let sub_k = &col_sub_indices[j];
        let sub_v = &col_sub_values[j];

        let mut s_diag = 0.0;
        for (idx, &k) in sub_k.iter().enumerate() {
            let l_kj = sub_v[idx];
            let val_kj = get_selected(&z_rows, &z_vals, j, k);
            s_diag += l_kj * val_kj;
        }
        set_selected(
            &z_rows,
            &mut z_vals,
            j,
            j,
            (1.0 / (l_jj * l_jj)) - (s_diag / l_jj),
        );

        let mut relevant_is = row_to_cols[j].clone();
        relevant_is.sort_by(|a, b| b.cmp(a));

        for i in relevant_is {
            let l_ii = col_diag[i];
            let sub_k_i = &col_sub_indices[i];
            let sub_v_i = &col_sub_values[i];

            let mut s_off = 0.0;
            for (idx, &k) in sub_k_i.iter().enumerate() {
                let l_ki = sub_v_i[idx];
                s_off += l_ki * get_selected(&z_rows, &z_vals, k, j);
            }
            set_selected(&z_rows, &mut z_vals, i, j, -s_off / l_ii);
        }
    }

    (z_rows, z_vals)
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

    let (z_rows, z_vals) = compute_takahashi(indptr, indices, data, n);

    // Flatten and Unpermute for return
    let selected_nnz: usize = z_rows.iter().map(Vec::len).sum();
    let mut out_rows = Vec::with_capacity(selected_nnz * 2);
    let mut out_cols = Vec::with_capacity(selected_nnz * 2);
    let mut out_vals = Vec::with_capacity(selected_nnz * 2);

    for j in 0..n {
        for (pos, &i) in z_rows[j].iter().enumerate() {
            let val = z_vals[j][pos];
            let orig_i = p_vec[i];
            let orig_j = p_vec[j];

            out_rows.push(orig_i);
            out_cols.push(orig_j);
            out_vals.push(val);

            if i != j {
                out_rows.push(orig_j);
                out_cols.push(orig_i);
                out_vals.push(val);
            }
        }
    }

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
    let (z_rows, z_vals) = compute_takahashi(
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

    let mut out_vals = Vec::with_capacity(rows.len());
    for (&row, &col) in rows.iter().zip(cols.iter()) {
        let factor_row = inv_perm[row as usize];
        let factor_col = inv_perm[col as usize];
        out_vals.push(get_selected(&z_rows, &z_vals, factor_row, factor_col));
    }

    out_vals.into_pyarray(py)
}

#[pymodule]
fn rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(takahashi_py, m)?)?;
    m.add_function(wrap_pyfunction!(takahashi_masked_py, m)?)?;
    Ok(())
}
