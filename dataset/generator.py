#!/usr/bin/env python3
import random
import csv

# This function recursively builds an arithmetic expression.
# The parameter num_ops indicates how many operations should be
# inserted (so num_ops==0 gives a single digit, and higher values
# produce longer expressions with nested operators).
def generate_expression(num_ops):
    # Base case: if no operators remain, return a single digit (0-9)
    if num_ops == 0:
        return str(random.randint(0, 9))

    # Decide how to split the remaining operations between the left and right parts.
    # We use one operator here, so subtract one operation.
    left_ops = random.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops

    # Recursively generate the left and right subexpressions.
    left_expr = generate_expression(left_ops)
    right_expr = generate_expression(right_ops)

    # Choose a random operator from the allowed list.
    op = random.choice(['+', '-', '*', '/'])

    # Build the expression with parentheses.
    expr = "(" + left_expr + op + right_expr + ")"
    return expr

# This function generates a specified number of expressions (with the given difficulty)
# and returns a list of (expression, result) tuples. If evaluation fails (for instance due to
# a division by zero), it will simply generate a new expression.
def generate_expressions(num_expressions, num_ops):
    expr_list = []
    while len(expr_list) < num_expressions:
        expr = generate_expression(num_ops)
        try:
            # Evaluate the expression.
            # Note: use eval with caution. In this controlled case we know that expr
            # contains only numbers, parentheses and operators.
            result = eval(expr)
        except ZeroDivisionError:
            continue  # skip and generate a new expression if division by zero occurred.
        except Exception as e:
            # Any other error: skip (or you could print/log the error)
            continue
        expr_list.append((expr, result))
    return expr_list

# This function writes a list of expression/result pairs to a CSV file.
def save_to_csv(expr_list, filename="expressions.csv"):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header.
        writer.writerow(["Expression", "Result"])
        # Write each expression and its result.
        for expr, result in expr_list:
            writer.writerow([expr, result])
    print(f"Saved {len(expr_list)} expressions to {filename}")

# MAIN
if __name__ == "__main__":
    # Adjust these parameters as needed:
    num_expressions = 20  # number of expressions to generate
    difficulty = 3       # number of operations; higher means a longer/more complex expression

    expressions = generate_expressions(num_expressions, difficulty)
    # Uncomment the next few lines to print out the generated expressions.
    # for e, r in expressions:
    #     print(f"{e} = {r}")
    save_to_csv(expressions)