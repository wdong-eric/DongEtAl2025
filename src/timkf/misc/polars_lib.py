import polars as pl


def cast_parenthetical_expr_to_valerr(
    df: pl.DataFrame,
    col: str,
):
    """
    Cast a column with parenthetical expressions to a value column and an error column.
    Some reminder of regular expressions used:
    - `[+-]?` matches an optional sign (either + or -)
    - `\d+` matches one or more digits
    - `\.?` matches an optional decimal point
    - `\d*` matches zero or more digits after the decimal point
    - `\((\d+)\)` captures digits inside parentheses
    - Note: `.` matches any character except a newline, so we use `\.` to match a literal dot.
    """
    df = df.with_columns(
        pl.col(col).str.extract(r"([+-]?\d+\.?\d*)").alias(f"{col} val"),
        pl.col(col).str.extract(r"\((\d+)\)").alias(f"{col} err"),
    )
    # cast the error in correct decimals based on the number of digits after the decimal point
    decimal_place_expr = (
        pl.when(pl.col(f"{col} val").str.contains(r"\."))
        .then(pl.col(f"{col} val").str.split(".").list.get(-1).str.len_chars())
        .otherwise(0)
    )
    return df.with_columns(
        pl.col(f"{col} err").cast(pl.Float64)
        / (10**decimal_place_expr).alias("Q error"),
        pl.col(f"{col} val").cast(pl.Float64).alias(f"{col} val"),
    )


def format_valerr_expr_to_latex(
    df: pl.DataFrame,
    med_expr: pl.Expr,
    upperr_expr: pl.Expr,
    lowerr_expr: pl.Expr,
    output_col: str,
    central_sigfigs: int = 2,
    error_sigfigs: int = 1,
) -> pl.DataFrame:
    """
    Format the median and error expressions to LaTeX format.
    """
    return df.with_columns(
        pl.format(
            "${}^{+{}}_{{}}$",
            med_expr.round_sig_figs(central_sigfigs),
            upperr_expr.round_sig_figs(error_sigfigs),
            lowerr_expr.round_sig_figs(error_sigfigs),
        ).alias(output_col)
    )


def format_valerr_to_latex(
    df: pl.DataFrame,
    med_col: str,
    err_col: str,
    output_col: str,
    log_scale: bool = True,
    central_sigfigs: int = 2,
    error_sigfigs: int = 1,
) -> pl.DataFrame:
    """
    Format the median and error columns to LaTeX format.
    """
    med_expr = pl.col(med_col)
    err_expr = pl.col(err_col)
    upp_expr = med_expr + err_expr
    low_expr = med_expr - err_expr
    if log_scale:
        med_expr = med_expr.log10()
        err_expr = err_expr.log10()
        upp_expr = upp_expr.log10()
        low_expr = low_expr.log10()

    return format_valerr_expr_to_latex(
        df,
        med_expr,
        upp_expr - med_expr,
        low_expr - med_expr,
        output_col,
        central_sigfigs=central_sigfigs,
        error_sigfigs=error_sigfigs,
    )
