import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

transformations = standard_transformations + (implicit_multiplication_application,)

def clean_latex_input(expr_str):
    """Normalize input strings into parseable SymPy format."""
    clean = expr_str.strip()
    clean = clean.replace("\\cdot", "*").replace("^", "**")
    clean = clean.replace("\\times", "*").replace(" ", "")
    clean = clean.replace("{", "(").replace("}", ")")
    return clean

def diagnose_step(lhs_prev, rhs_prev, lhs_curr, rhs_curr, var=sp.Symbol('x')):
    """
    Analyzes common algebraic misconceptions and derives the correct next step.
    Returns: (diagnostic_feedback, correct_next_step_latex)
    """
    try:
        prev_eq = sp.Eq(lhs_prev, rhs_prev)
        sol_prev = sp.solve(prev_eq, var)
        
        curr_eq = sp.Eq(lhs_curr, rhs_curr)
        sol_curr = sp.solve(curr_eq, var)

        if sol_prev:
            correct_val = sol_prev[0]
            correct_step_latex = f"{sp.latex(var)} = {sp.latex(correct_val)}"
        else:
            correct_step_latex = sp.latex(sp.simplify(lhs_prev - rhs_prev)) + " = 0"

        if not sol_prev or not sol_curr:
            return "Algebraic transformation error detected.", correct_step_latex

        student_val = sol_curr[0]
        correct_val = sol_prev[0]

        coeff = lhs_prev.coeff(var) if lhs_prev.has(var) else rhs_prev.coeff(var)
        const = rhs_prev if lhs_prev.has(var) else lhs_prev

        if coeff and const:
            if sp.simplify(student_val - (const * coeff)) == 0:
                msg = f"Multiplication/Division Error: You multiplied by {coeff} instead of dividing both sides by {coeff}."
                return msg, correct_step_latex

            if sp.simplify(student_val - (const - coeff)) == 0:
                msg = f"Transposition Error: You subtracted coefficient {coeff} instead of dividing both sides by {coeff}."
                return msg, correct_step_latex

            if sp.simplify(student_val + correct_val) == 0:
                msg = "Sign Inversion Error: Watch the sign change when moving terms across the equality sign."
                return msg, correct_step_latex

        return "Algebraic mismatch: The step does not preserve mathematical equivalence.", correct_step_latex

    except Exception:
        return "Algebraic error detected: Verify arithmetic and balance across equality.", None

def verify_step(prev_expr_str, curr_expr_str):
    """
    Verifies step correctness and returns LaTeX representations.
    Returns: (is_valid, message, latex_prev, latex_curr, latex_correction)
    """
    try:
        clean_prev = clean_latex_input(prev_expr_str)
        clean_curr = clean_latex_input(curr_expr_str)

        var = sp.Symbol('x')
        if 'y' in clean_prev and 'x' not in clean_prev:
            var = sp.Symbol('y')

        if "=" in clean_prev and "=" in clean_curr:
            p_parts = clean_prev.split("=")
            c_parts = clean_curr.split("=")

            lhs_prev = parse_expr(p_parts[0], transformations=transformations)
            rhs_prev = parse_expr(p_parts[1], transformations=transformations)
            lhs_curr = parse_expr(c_parts[0], transformations=transformations)
            rhs_curr = parse_expr(c_parts[1], transformations=transformations)

            latex_prev = f"{sp.latex(lhs_prev)} = {sp.latex(rhs_prev)}"
            latex_curr = f"{sp.latex(lhs_curr)} = {sp.latex(rhs_curr)}"

            diff_prev = sp.simplify(lhs_prev - rhs_prev)
            diff_curr = sp.simplify(lhs_curr - rhs_curr)

            if sp.simplify(diff_prev - diff_curr) == 0 or sp.simplify(diff_prev + diff_curr) == 0:
                return True, "Valid algebraic deduction. Both sides remain balanced.", latex_prev, latex_curr, None

            ratio = sp.simplify(diff_prev / diff_curr)
            if ratio.is_number and ratio != 0:
                return True, "Valid algebraic deduction (scalar transformation).", latex_prev, latex_curr, None

            diag_msg, correct_latex = diagnose_step(lhs_prev, rhs_prev, lhs_curr, rhs_curr, var=var)
            return False, diag_msg, latex_prev, latex_curr, correct_latex

        else:
            e1 = parse_expr(clean_prev, transformations=transformations)
            e2 = parse_expr(clean_curr, transformations=transformations)
            latex_prev = sp.latex(e1)
            latex_curr = sp.latex(e2)

            if sp.simplify(e1 - e2) == 0:
                return True, "Valid algebraic simplification.", latex_prev, latex_curr, None
            
            correct_latex = sp.latex(sp.simplify(e1))
            return False, "Expressions are not algebraically equivalent.", latex_prev, latex_curr, correct_latex

    except Exception as e:
        return False, f"Syntax error: Could not parse expression ({str(e)})", None, None, None