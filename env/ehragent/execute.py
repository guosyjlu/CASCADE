import traceback
from func_timeout import func_timeout, FunctionTimedOut

CodeHeader = """import env.ehragent.tabtools as tabtools
LoadDB = tabtools.db_loader
FilterDB = tabtools.data_filter
GetValue = tabtools.get_value
SQLInterpreter = tabtools.sql_interpreter
Calendar = tabtools.date_calculator
"""

def run_code(cell):
    """
    Returns the path to the python interpreter.
    """
    try:
        global_var = {"answer": None}
        if 'answer' not in cell:
            return False, "Please save the answer to the question in the variable 'answer'."
        exec(CodeHeader+cell, global_var)
        return True, str(global_var['answer'])
    except Exception as e:
        error_info = traceback.format_exc().strip().splitlines()[-1]
        if len(error_info) >= 800:
            error_info = error_info[:800]
        return False, error_info


def run_code_with_limited_time(cell, timeout=60):
    try:
        return func_timeout(timeout, run_code, args=(cell,))
    except FunctionTimedOut:
        return False, f"Error: Code execution exceeded the time limit of {timeout} seconds."
    except Exception as e:
        return False, f"Error: {str(e)}"