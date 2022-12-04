## Blossom LP Function Draft -
# Purpose: Alg. 1 of Paper. Uses LP to solve min-cost perfect matching problem. Coined Blossom LP.
# Input: graph G and maximum number of iterations (optional, default is 2 times number of vertices) 
# Output: edges of min-cost perfect matching. If no perfect matching, return empty set.
#####
# Notes: 
# - Built on Networkx, however code can possibly be generalized.
#      - Instead of graph G a preferred input is arrays of vertices, edges, incidence matrix, and edge weights.
# - Graph update steps, contraction and expansion, via incidence matrix manipulation and not modifying graph structure.

# - Main Performance Issue: LP solver takes up about more than half of an iteration runtime (between 50% to 95%). Unreliable results for large graphs.  

# - Other performance improvements can be made: 
#      - Better selection and handling of data structures.
#      - Better and fewer search operations.
#      - Currently finds cycle edges via matrix multiplication and search. Maybe better method.


def BlossomLP(G, max_iterations = None):
    tol = 1e-09 # value comparison tolerance - value equal to default tolerance in math.isclose function    
    
    start_init = timeit.default_timer() ##### Test initialize speed
    
    # Initialize max_iterations
    if max_iterations == None:
        max_iterations = 2*G.number_of_nodes()
    
    # Initialize weights, Z, and blossoms
    weights = []
    for node1, node2, data in G.edges(data=True):
        weights.append(data['weight'])
    weights_adj = np.asarray(weights) # objective function parameters - weights (future versions of algorithm should try to perturb weights)
    Z = {} # dictionary: shrunk vertices -> weight adjustments
    blossoms = [] # list: blossom vertices lists
    IM = nx.incidence_matrix(G) # Incidence matrix of G
    
    # Initialize constraint matrices
    A_orig = IM.toarray()
    A_orig = A_orig[A_orig.any(axis=1)] # remove row of 0s
    num_constraints_orig = np.size(A_orig,0) # number of initial contraints for LP
    A_pseudo = np.empty(shape = [0, G.number_of_edges()])
    
    # Initialize M
    M = np.asarray([True]*num_constraints_orig) # True if vertex i in G'
    
    # Initialize min-cost perfect matching edges
    matched_edges = np.array([])
    
    # Check if even number of vertices to match
    if num_constraints_orig%2:
        print("\n\nOdd number of vertices to match, n = {}. Therefore no perfect matching.".format(num_constraints_orig))
        return matched_edges # Return empty set

    print("Initialization speed: ", timeit.default_timer() - start_init) #####
    
    iter_num = 1
    ### While loop Start ###
    while True:
        print("\n\nIteration: ", iter_num)
        print("weights =", weights_adj)
        
        start_iter = timeit.default_timer() ##### test iteration speed
        
        # Inequality constraints
        A_o = A_orig[np.where(M[:num_constraints_orig] == True)[0]]
        b_o = np.ones(np.size(A_o,0))
        A_p = A_pseudo[np.where(M[num_constraints_orig:] == True)]
        b_p = np.ones(np.size(A_p,0))

        # Solve LP - better LP solver?
        
        start_LP = timeit.default_timer() ### Performance measure - LP
        
        res_BLP = linprog(weights_adj, A_eq = A_o, b_eq = b_o, A_ub = -A_p, b_ub = -b_p, method = 'revised simplex')
        
        
        speed_LP = timeit.default_timer() - start_LP
        print("LP performance speed: ", speed_LP)
        print(res_BLP)
        
        # Check if LP has solution
        if not res_BLP.success:
            print("\n\nNo perfect matching.")
            break # Return empty set
        
        # Check if LP solution unique. i.e. no contraction-expansion loops and alg. terminates.
        if iter_num > max_iterations:
            print("\n\nLP solution not unique. Try perturbing weights or changing contraction order.")
            break # Return empty set
            
        ## Check if expand else contract
        pseudo_matchings = np.matmul(A_p,res_BLP.x) # matchings per pseudo vertex
        isin_cycle = np.isclose(0.5, res_BLP.x) # check if matching edge in cycle
        
        # Expansion        
        if (pseudo_matchings>1+tol).any(): # Check if claw exists, if so expand pseudo vertex
            start_exp = timeit.default_timer() ##### test expansion speed

            print("Expansion...")
            
            if (pseudo_matchings>2+tol).any():
                print("LP solution not unique. Restarting algorithm...")     
                # Reinitialize parameters
                A_pseudo = np.empty(shape = [0, G.number_of_edges()])
                weights_adj = np.asarray(weights) # objective function parameters - weights
                Z = {} # dictionary: shrunk vertices -> weight adjustments
                blossoms = [] # list: blossom vertices lists
                M = np.asarray([True]*num_constraints_orig) # True if vertex i in G'
                continue
                
            
            blossoms_filter = np.matmul(A_pseudo,res_BLP.x)>1+tol
            S_i = np.where(blossoms_filter)[0][0] # Blossom vertex index
            for v_i in blossoms[S_i]:
                for e_i in np.where(np.append(A_orig, A_pseudo, axis = 0)[v_i,:] + A_pseudo[S_i] == 2)[0]: # Do I need tolerance? I don't think so, incidence matrix should be type integer.
                    weights_adj[e_i] += Z[v_i]  # Update weights
                M[v_i] = True # vertex in G'
            M[num_constraints_orig + S_i] = False # pseudo vertex not in G' 
            
            print("Expansion speed performance: ", timeit.default_timer() - start_exp) #####
            
        # Contraction        
        elif isin_cycle.any(): 
            start_con = timeit.default_timer() ##### test contraction speed

            print("Contraction...") # If no claw and no perfect matching, then a cycle exists.
            
            # Tracking Cycles in script C, see paper, and picking a cycle stage.
            C = np.array([])
            cycles_G = np.where(isin_cycle)[0] 
            
            # Update cycles in script C
            if iter_num == 1:
                cycles = cycles_G # initialize cycles in script C
            else:
                cycles = np.intersect1d(cycles, cycles_G, assume_unique=True) # update cycles in script C
                C = np.setdiff1d(cycles_G, cycles, assume_unique=True) # new cycle not in script C
                
            if C.size == 0: # No new cycle
                # Pick random cycle in script C
                k = rand.randint(0, cycles.size-1)
                C = np.array(cycles[k]) # initialize cycle C by choosing arbitrary edge in script C

                # Find rest of edges in cycle C above.
                # Computes edge adjacency matrix between C (row) and script C (column). Adds edge from script C to C if column entries sum to 1, else terminate while loop.
                # Since odd cycle, two edges are always added to C and last two edges added are always adjacent.
                while True: 
                    if C.size == 1:
                        temp = np.matmul(A_orig[:, C], A_orig[:, cycles]) # C-script C edge adjacency matrix 
                        temp_edges = cycles[np.where(temp == 1)] # if edge adjacency matrix entry equals 1 then edge from script C to add to C.
                    else:
                        temp = np.matmul(np.transpose(A_orig[:, C]), A_orig[:, cycles]) # C-script C edge adjacency matrix 
                        temp_edges = cycles[np.where(temp.sum(axis=0) == 1)] # if edge adjancency matrix column sums to 1, then edge from script C to add to C.
                        
                    C = np.append(C, temp_edges) # add edges to C
                    if A_orig[:,temp_edges].all(axis=1).any(): # Check if newly added edges are adjacent.
                        break

                # Remove chosen cycle C from script C
                np.setdiff1d(cycles, C, assume_unique=True)
        
            
            print("Contracted edges: ", np.asarray(G.edges)[C])
            
            # Build blossom and update pseudo vertex incidence matrix
            blossoms.append(np.where(M & np.any(np.append(A_orig, A_pseudo, axis = 0)[:, C], axis=1))[0]) # add blossom vertex
            # Update pseudo vertex incidence matrix
            S = np.append(A_orig, A_pseudo, axis = 0)[blossoms[-1]].sum(axis = 0)
            S[S > 1] = 0 # remove cycle edge incidences
            A_pseudo = np.append(A_pseudo, [S], axis=0)
            
            # calculate z values and save them in Z
            z_val = np.linalg.solve(np.append(A_orig, A_pseudo, axis = 0)[:, C][blossoms[-1], :], weights_adj[C])
            
            for i in range(0, len(blossoms[-1])):
                v_i = blossoms[-1][i]
                # save in Z
                Z[v_i] = z_val[i]
                # Update weights given Z and update M
                for e_i in np.where(np.append(A_orig, A_pseudo, axis = 0)[v_i,:] + S == 2)[0]:
                    weights_adj[e_i] -= z_val[i]  
                M[v_i] = False # vertex not in G'
            M = np.append(M, True) # new pseudo vertex in G'
            
            print("Contraction speed performance: ", timeit.default_timer() - start_con) #####
            
        else:
            start_match = timeit.default_timer() ##### test matching speed
            
            print("\n\nFound perfect matching.")
            perfect_matching = res_BLP.x
            print("Perfect matching with blossoms: ", np.asarray(G.edges)[np.where(np.isclose(1, res_BLP.x))])

            # Check for blossoms
            if not M[:num_constraints_orig].all():
                # Expand blossoms and run LP on blossom original vertices excluding matched vertex.
                unmatched_v = np.where(M[:num_constraints_orig] == False)[0] # select blossom nodes
                unmatched_v = unmatched_v[np.where(np.isclose(0, np.matmul(A_orig[unmatched_v, :], res_BLP.x)))] # remove matched vertices
                unmatched_e = np.where([e.all() for e in np.isin(np.array(list(G.edges())), np.asarray(G.nodes)[unmatched_v])])[0] # select unmatched edges in blossom
                # Match unmatched edges in blossom via LP
                unmatched_BLP = linprog(weights_adj[unmatched_e], A_eq = A_orig[unmatched_v, :][:, unmatched_e], b_eq = np.ones(len(unmatched_v)), method = 'revised simplex')
                matched_e = unmatched_e[np.where(np.isclose(1, unmatched_BLP.x))] # indices of newly matched edges
                perfect_matching[matched_e] = 1 # update perfect matching with matched blossom edges

            # Return set of edges for perfect matching. Check for errors.
            check = np.all(np.isclose(1, np.matmul(A_orig, perfect_matching))) # check that each vertex is incident to one edge
            if check:
                matched_edges = np.asarray(G.edges)[np.where(np.isclose(1, perfect_matching))] # edges of matching
                print("Perfect matching incidence vector: ", perfect_matching)
                print("Perfect matching edges: ", matched_edges)
                print("Number of iterations: ", iter_num)
            else:
                print("Error. Not a perfect matching. Need to debug!") ## Error message
            
            
            print("matching performance speed: ", timeit.default_timer() - start_match) #####
            speed_iter = timeit.default_timer() - start_iter
            print("iteration performance speed: ", speed_iter) #####
            print("% solve LP: {:.2f}%".format(100*speed_LP/speed_iter) )
            
            break

        # Update iteration number
        iter_num += 1
        
         
        speed_iter = timeit.default_timer() - start_iter
        print("iteration performance speed: ", speed_iter) #####
        print("% solve LP: {:.2f}%".format(100*speed_LP/speed_iter) )
        
        matched_edges = np.asarray(G.edges)[np.where(np.isclose(1, res_BLP.x))] # edges of matching
        print("G' matched edges: ", matched_edges)
       
    return matched_edges
