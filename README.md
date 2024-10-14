# Theory of Computing - DPLLSolver
Theory of Computing Project 1 - 2SAT Solver using the DPLL Algorithm

Much inspiration for the program came from the pseudocode found on Wikipedia for the DPLL Algorithm:
https://en.wikipedia.org/wiki/DPLL_algorithm.  

The DPLL algorithm is the basis for most modern SAT solvers. Its worst case time complexity is O(2n) where n is the number of 
variables in the formula. The DPLL is an improvement for other solvers as it uses backtracking and unit propagation. Unit propagation consists in removing every clause containing a unit clause's literal and in discarding the complement of a unit clause's literal from every clause containing that complement. Bactracking works by choosing a literal, assigning a truth value to it, simplifying the formula and then recursively checking if the simplified formula is satisfiable. If not satsifiable it tries the opposite truth value.

# Here is the Wikipedia pseudocode:
Algorithm DPLL. 
    Input: A set of clauses Φ.  
    Output: A truth value indicating whether Φ is satisfiable.  
function DPLL(Φ). 
    // unit propagation:  
    while there is a unit clause {l} in Φ do. 
        Φ ← unit-propagate(l, Φ);  
    // pure literal elimination:  
    while there is a literal l that occurs pure in Φ do. 
        Φ ← pure-literal-assign(l, Φ);  
    // stopping conditions:  
    if Φ is empty then. 
        return true;  
    if Φ contains an empty clause then. 
        return false;  
    // DPLL procedure:  
    l ← choose-literal(Φ);  
    return DPLL(Φ ∧ {l}) or DPLL(Φ ∧ {¬l});  


