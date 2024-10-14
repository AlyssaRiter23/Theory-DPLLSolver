import random
import time
import matplotlib.pyplot as plt
import numpy as np
import copy

# DPLL algorithm for 2-SAT in Python
def DPLLAlgorithm(cnf, assignment={}):
    # unit propagation -> check if there is a unit clause
    # if there is a unit clause it must be set to true
    while has_unit_clause(cnf):
        l = get_unit_clause(cnf)
        cnf = unit_propagation(l, cnf)
        if has_empty_clause(cnf):  # conflict detected during propagation
            return False, {}
        # unit clause must be true
        assignment[l] = True

    # pure literal elimination
    # a literal is pure if its negation is not found in the cnf
    # set that literal to true and check if there is an empty clause
    while has_pure_literal(cnf):
        l = get_pure_literal(cnf)
        if l is not None:
            cnf = pure_literal_assign(l, cnf, assignment)
        if has_empty_clause(cnf):  # conflict has been found while trying to detect pure lits
            return False, {}

    # base case/stopping conditions
    # cnf has no clauses
    if is_empty(cnf):
        return True, assignment
    # clause contains no literals
    if has_empty_clause(cnf):
        return False, {}

    # choose a literal based on how often it appears in the cnf
    l = choose_literal(cnf)

    # recursive calls for DPLL alg -> 2 of them: one for true, other for false
    # updating the assingment with literal equal to true
    sat, updated_assignment = DPLLAlgorithm(unit_propagation(l, copy_cnf(cnf)), {**assignment, l: True})
    if sat:
        return True, updated_assignment
    # updating the assignment with literal equal to False and passing in negation of literal
    sat, updated_assignment = DPLLAlgorithm(unit_propagation(-l, copy_cnf(cnf)), {**assignment, l: False})
    return sat, updated_assignment

# check if the cnf contains a unit clause (only one literal)
def has_unit_clause(cnf):
    for clause in cnf:
        if len(clause) == 1:
            return True
    return False

# get the first unit clause
def get_unit_clause(cnf):
   for clause in cnf:
    if len(clause) == 1:
        return clause[0]

# unit propagation for a literal
def unit_propagation(literal, cnf):
    # create a new cnf
    new_cnf = []
    for clause in cnf:
        # remove the clauses where the literal is satisfied
        if literal in clause:
            continue  
        # remove the negations of the literals
        new_clause = [lit for lit in clause if lit != -literal]  # Remove negation of the literal
        if not new_clause:  # empty clause means conflict
            return [[]]  # return empty cnf to indicate conflict
        new_cnf.append(new_clause)
    return new_cnf

# check if the cnf has a pure literal
def has_pure_literal(cnf):
    # use a dictionary to keep track of the number of each type of literal
    literals = {}
    for clause in cnf:
        # count the number of times a literal occurs
        for literal in clause:
            # update dictionary count
            if literal in literals:
                literals[literal] += 1
            else:
                literals[literal] = 1
    # check if its negation is in literals
    for literal in literals:
        if -literal not in literals:
            return True
    return False

# get a pure literal -> means negation is not in the cnf
def get_pure_literal(cnf):
    # dictionary to keep track of literal counts
    literals = {}
    for clause in cnf:
        for literal in clause:
            # if in dictionary increment count by 1 otherwise count is 1
            if literal in literals:
                literals[literal] += 1
            else:
                literals[literal] = 1
    for literal in literals:
        if -literal not in literals:
            return literal
    return None

# assign pure literals by removing clauses containing them
def pure_literal_assign(literal, cnf, assignment):
    if literal > 0:
        assignment[literal] = True
    else: 
        assignment[-literal] = False
    new_cnf = []
    for clause in cnf:
        if literal in clause:
            continue
        new_clause = [lit for lit in clause if lit != literal]
        new_cnf.append(new_clause)
    return new_cnf#[clause for clause in cnf if literal not in clause]

# check if the cnf is empty (satisfiable)
def is_empty(cnf):
    return len(cnf) == 0

# check if the cnf contains an empty clause (unsatisfiable)
def has_empty_clause(cnf):
    for clause in cnf:
        if len(clause) == 0:
            return True
    return False
    

# choose a literal based on how many times it occurs in the cnf (for efficiency)
def choose_literal(cnf):
    # use a dictionary again for the literal counter to find most frequent literal
    literal_counter = {}
    for clause in cnf:
        for literal in clause:
            if literal in literal_counter:
                literal_counter[literal] += 1
            else:
                literal_counter[literal] = 1
    most_freq = max(literal_counter, key= lambda x: literal_counter[x])
    return most_freq  # choose the most frequent literal -> check if there is another way to write this

# copy the cnf so that it doesn't get overwritten in the recursive calls
def copy_cnf(cnf):
    copied_cnf = []
    for clause in cnf:
        copied_cnf.append(clause)
    return copy.deepcopy(cnf)


# generate random 2-SAT formulas with 2 literals per clause
def generate_random_2sat(num_variables, num_clauses):
    cnf = set()  # set -> avoids duplicate clauses
    while len(cnf) < num_clauses:
        clause = []
        # 2-SAT cnf
        while len(clause) < 2:
            # generate a literal with a random number
            literal = random.randint(1, num_variables*2)
            if random.choice([True, False]):  # decides if true or false (negation)
                literal = -literal
            if literal not in clause and -literal not in clause:  # avoids duplicates
                clause.append(literal)
        cnf.add(tuple(clause))  # make the clauses tuples 
    
     # write the generated CNF to a file
    with open("random_cnfs.txt", "a") as cnf_file:
        cnf_file.write(f"CNF with {num_variables} variables and {num_clauses} clauses:\n")
        for clause in cnf:
            cnf_file.write(f"{clause}\n")
        cnf_file.write("\n")

    return list(cnf) # return a list of tuples (2-SATS)

# test the DPLL algorithm on random 2-SAT CNF instances
def test_dpll_polynomial(trials, num_variables_range):
    # intialize lists for graphing!!
    execution_times_sat = []
    execution_times_unsat = []
    variables_list_sat = []
    variables_list_unsat = []
    
    with open("resultsfile.txt", "a") as f:
        for num_variables in num_variables_range:
            num_clauses = 3 * num_variables  # using 3 * variables as a standard case for the number of clauses
            for trial in range(trials):
                # generate a random 2-SAT CNF formula
                cnf = generate_random_2sat(num_variables, num_clauses)

                # measure time for DPLL algorithm
                start_time = time.time()
                sat, assignment = DPLLAlgorithm(cnf)  
                end_time = time.time()
                # calculate the execution time
                execution_time = end_time - start_time
                # if satisfiable append to sat lists 
                if sat:
                    variables_list_sat.append(num_variables)
                    execution_times_sat.append(execution_time)
                    f.write(f"Satisfiable - Variables: {num_variables}, Clauses: {num_clauses}, Assignment: {assignment}, Time: {execution_time:.6f}s\n")
                else:
                    variables_list_unsat.append(num_variables)
                    execution_times_unsat.append(execution_time)
                    f.write(f"Unsatisfiable - Variables: {num_variables}, Clauses: {num_clauses}, Time: {execution_time:.6f}s\n")
            # print metrics
            print(f"Variables: {num_variables}, Clauses: {num_clauses}, Assignment: {assignment}\n")
    # return all the metrics
    return (variables_list_sat, execution_times_sat, 
            variables_list_unsat, execution_times_unsat)

# plotting the results using python's scatter plot from matplotlib
def plot_execution_times(variables_list_sat, execution_times_sat, variables_list_unsat, execution_times_unsat):
    print("running plots") # for debugging
    coeffs = np.polyfit(variables_list_unsat, execution_times_unsat, 2)  # You can change the degree
    poly = np.poly1d(coeffs)
    plt.scatter(variables_list_sat, execution_times_sat, color='green', label="Satisfiable CNFs", s=40)
    plt.scatter(variables_list_unsat, execution_times_unsat, color='red', label="Unsatisfiable CNFs", s = 40)
    plt.plot(variables_list_unsat, poly(variables_list_unsat), label='Polynomial Fit', linestyle='--')
    #plt.plot(num_variables_list, execution_times, label="Execution Time vs Variables")
    plt.xlabel("Number of Variables")
    plt.ylabel("Execution Time (seconds)")
    plt.title("2-SAT DPLL Execution Time")
    plt.legend()
    plt.grid(True)
    # Display the equation of the polynomial
    equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
    plt.text(0.5, 0.5, equation, horizontalalignment='right', verticalalignment='bottom', 
         transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))   
    # save as an image since ssh machines don't allow for python graph pop-ups
    plt.savefig("execution_time_plot.png")
    print("plot shown") # for debugging

# arguments for testing
NUM_TRIALS = 10  # number of trials per test case
VARIABLE_RANGE = range(3, 100, 5)  # range of variables to test (2 to 50 variables)
# run tests and plot the results
variables_list_sat, execution_times_sat, variables_list_unsat, execution_times_unsat = test_dpll_polynomial(NUM_TRIALS, VARIABLE_RANGE)
plot_execution_times(variables_list_sat, execution_times_sat, variables_list_unsat, execution_times_unsat)
