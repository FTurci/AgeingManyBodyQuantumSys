module NM_measures_1body

# Export all relevant functions and types so they are accessible in your notebook
export base_params, set_P, spectral_function, thermofield_double, 
       chain_map, H_bath, H_tot, prepare_corrs, evolve_corrs, 
       spin_operators, matrix_operators, matrix_log, map_to_principal, 
       ρ_to_Λ, Λ_to_ρ, calculate_ρ_using_G, calculate_RHP, calculate_BLP, 
       calculate_measures

using LinearAlgebra
using PolyChaos
using QuadGK
using Plots
using ProgressMeter
using Kronecker
using JLD2
using SparseArrays

mutable struct base_params
    spec_fun ::AbstractString        #type of spectral function
    N_L ::Int64                      #Number of left bath sites
    N_R ::Int64                      #Number of right bath sites
    β_L ::Float64                    #inverse temperature of left bath
    β_R ::Float64                    #inverse temperature of right bath
    μ_L ::Float64                    #chemical potential of left Bath
    μ_R ::Float64                    #chemical potential of right Bath
    D_L ::Float64                    #left bath bandwidth
    D_R ::Float64                    #Right bath bandwidth
    Γ_L ::Float64                    #overall left bath coupling strength
    Γ_R ::Float64                    #overall right bath coupling strength
    E_sys ::Float64                  #system site energy
    dt ::Float64                     #timestep length
    init_occ ::Float64               #initial occupation

    base_params() = new()
end

function set_P()
    """initialises the base params P to some default settings"""
    P = base_params();
    P.spec_fun = "elliptical"
    P.N_L = 100
    P.N_R = 100
    P.β_L = 20.0
    P.β_R = 20.0
    P.μ_L = 0.0
    P.μ_R = 0.0
    P.D_L = 1.0
    P.D_R = 1.0
    P.Γ_L = 0.01
    P.Γ_R = 0.01
    P.E_sys = 0.0
    P.dt = 0.1
    return P
end

function spectral_function(input::AbstractString, D::Float64, g::Float64)
    """creates spectral functio(s) for a bath"""
    inband = x -> (-D <= x <= D)
    
    if input == "flat"
        J = x -> inband(x) ? 1/(2D) : 0.0
    elseif input == "elliptical"
        J = x -> inband(x) ? sqrt(1 - (x/D)^2) : 0.0
    elseif input == "ohmic"
        J = x -> inband(x) ? abs(x) : 0.0
    elseif input == "lorentzian"
        J = x -> 1/(1 + (x/D)^2)
    else
        error("spectral function type not recognized")
    end
    # normalization
    norm = quadgk(J, -D, D)[1]           
    Jnorm = x -> g*D/pi * J(x) / norm
    return Jnorm
end

function thermofield_double(J, beta::Float64, mu::Float64)
    """thermofield purification using fermi function ancilla"""
    fermi(k) = 1/(1 + exp(beta*k - beta*mu))
    J1 = w -> J(w) * fermi(w) #filled mode spectral density
    J2 = w -> J(w) * (1 - fermi(w)) #empty mode spectral density
    return J1, J2
end

function chain_map(J, N::Int64, D::Float64)
    """calculates family of monic orthogonal polynomials w.r.t the measure J(x) up to the Nth term."""
    supp = (-D, D)
    meas = Measure("bath", J, supp, false, Dict())
    ortho_poly = OrthoPoly("bath_op", N, meas; Nquad=100000)   
    chain = coeffs(ortho_poly)                                  
    E = chain[1:N,1] #site energies
    h = sqrt.(chain[1:N,2]) #site hoppings
    return E, h
end

function H_bath(N::Int64, J, D::Float64, beta::Float64, mu::Float64)
    """makes Hamiltonian for single bath in chain basis, interleaved filled and empty chain"""
    H = zeros(ComplexF64, 2N, 2N)
    J1, J2 = thermofield_double(J, beta, mu)
    E1, h1 = chain_map(J1, N, D) #filled chain
    E2, h2 = chain_map(J2, N, D) #empty chain

    g1 = h1[1] #filled chain to system coupling
    g2 = h2[1] #empty chain to system coupling

    t1 = h1[2:end] #chain hoppings
    t2 = h2[2:end] 

    for n in 1:N
        H[2n-1, 2n-1] = E1[n]
        H[2n, 2n] = E2[n]
    end
    
    for n in 1:N-1
        H[2n-1,2n+1] = t1[n]
        H[2n+1,2n-1] = t1[n]
        H[2n, 2n+2] = t2[n]
        H[2n+2, 2n] = t2[n]
    end

    return H, g1, g2
end

function H_tot(J_L, J_R, P)
    """full Hamiltonian for baths + system + ancilla"""
    (;N_L, N_R, β_L, β_R, μ_L, μ_R, D_L, D_R, E_sys) = P

    if N_L == 0 
        H_R, g1_R, g2_R = H_bath(N_R, J_R, D_R, β_R, μ_R)
        H_sys = [0.0 0.0 ; 0.0 E_sys] 
        H = cat(H_sys, H_R; dims=(1,2))
        H[2, 3] = g1_R
        H[3, 2] = g1_R
        H[2, 4] = g2_R
        H[4, 2] = g2_R
        return Matrix(H)
    end

    H_L, g1_L, g2_L = H_bath(N_L, J_L, D_L, β_L, μ_L)
    H_R, g1_R, g2_R = H_bath(N_R, J_R, D_R, β_R, μ_R)
    H_sys = [0.0 0.0 ; 0.0 E_sys]
    
    H = cat(reverse(H_L), H_sys, H_R; dims=(1,2))

    # Connection indices
    sys_idx = 2*N_L + 2
    
    H[sys_idx, 2N_L] = g1_L
    H[2N_L, sys_idx] = g1_L
    H[sys_idx, 2N_L - 1] = g2_L
    H[2N_L - 1, sys_idx] = g2_L

    H[sys_idx, 2N_L + 3] = g1_R
    H[2N_L + 3, sys_idx] = g1_R
    H[sys_idx, 2N_L + 4] = g2_R
    H[2N_L + 4, sys_idx] = g2_R

    return Matrix(H)
end

function prepare_corrs(P)
    """creates initial correlation matrix"""
    (;N_L, N_R) = P

    function make_bath_corr(N)
        C = zeros(ComplexF64, 2N, 2N)
        for n in 1:2N
            if isodd(n) 
                C[n,n] = 1.0 
            else 
                C[n,n] = 0.0 
            end
        end
        return C
    end

    if N_L == 0
        C_R = make_bath_corr(N_R)
        C_emp = complex([0.0 0.0 ; 0.0 0.0]) 
        C_full = complex([0.0 0.0 ; 0.0 1.0])
        C_CJ = complex([0.5 0.5 ; 0.5 0.5]) 

        C0_emp = cat(C_emp, C_R; dims=(1,2))
        C0_full = cat(C_full, C_R; dims=(1,2))
        C0_CJ = cat(C_CJ, C_R; dims=(1,2))
        return C0_emp, C0_full, C0_CJ
    end

    C_L = make_bath_corr(N_L)
    C_R = make_bath_corr(N_R)

    C_emp = complex([0.0 0.0 ; 0.0 0.0]) 
    C_full = complex([0.0 0.0 ; 0.0 1.0]) 
    C_CJ = complex([0.5 0.5 ; 0.5 0.5]) 

    C0_emp = cat(reverse(C_L), C_emp, C_R; dims=(1,2))
    C0_full = cat(reverse(C_L), C_full, C_R; dims=(1,2))
    C0_CJ = cat(reverse(C_L), C_CJ, C_R; dims=(1,2))

    return C0_emp, C0_full, C0_CJ
end

function evolve_corrs(C0, H, P)
    (;N_L, N_R, D_L, D_R, dt) = P
    Cs = Vector{Array{ComplexF64}}(undef, 0)
    C_curr = Matrix(C0)
    H = Matrix(H)
    
    step_min = 500
    step_max = 2000
    if N_L >= N_R
        step_max = round(Int, 1.5*N_L/(dt*max(D_L, D_R))) 
    else
        step_max = round(Int, 1.5*N_R/(dt*max(D_L, D_R)))
    end
    
    U_dt = exp(-im * dt * H) 
    U_dt_dag = U_dt'
    
    push!(Cs, C_curr)
    
    step = 1
    settled = false
    
    while !settled
        C_curr = U_dt * C_curr * U_dt_dag
        push!(Cs, C_curr)
        
        if (step % 100 == 0) && (step > step_min)
            if length(Cs) > 100
                sys_idx = (N_L > 0) ? 2*N_L + 2 : 2
                nSys = [real(M[sys_idx, sys_idx]) for M in Cs[end-100:end]]
                min_val, max_val = extrema(nSys)
                settled = (max_val - min_val < 1e-5)
            end
        end
        step += 1
        if step > step_max; break; end 
    end
    return Cs
end

# ==============================================================================
# Dynamical Map Extraction & RHP Calculation Functions
# ==============================================================================

function spin_operators(M)
    sp = spdiagm(2,2,1=>ones(1))
    sm = spdiagm(2,2,-1=>ones(1))
    sz = spdiagm(2,2,0=>[1;-1]);
    num = spdiagm(2,2,0=>[0;1])
    Sz = Vector{Any}(undef, M)
    Sp = Vector{Any}(undef, M)
    Sm = Vector{Any}(undef, M)
    Num = Vector{Any}(undef,M)
    for m=1:M
        Sz[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sz),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sp[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sp),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sm[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sm),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Num[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),num),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
    end
    return Sz,Sp,Sm,Num
end

function JW_string_mat(Sz,site,M)
    Z = 1.0*Matrix(I, 2^M, 2^M)
    for q=1:(site-1)
        Z = Z*Sz[q];
    end
    return Z
end

function matrix_operators(M)
    Sz,Sp,Sm,_ = spin_operators(M)
    cdag_mat = Vector{Any}(undef,M)
    c_mat = Vector{Any}(undef,M)
    for n=1:M
        Z = JW_string_mat(Sz,n,M)
        cdag_mat[n] = Z*Sm[n]
        c_mat[n]  = Z*Sp[n]
    end
    return cdag_mat,c_mat
end

function map_to_principal(z)
    im_ = imag(z)
    im_ = im_ - 2*π*floor((im_ + π)/(2*π))
    return Complex(real(z), im_)
end

function matrix_log(A)
    F = eigen(A)
    D = Diagonal(F.values)
    V = F.vectors
    V_inv = inv(V)
    principal_log_D = map_to_principal.(log.(Complex.(D)))
    log_A = V*principal_log_D*V_inv
    return log_A
end

function ρ_to_Λ(ρ, Ns)
    """Rearranges the Choi state ρ to give the Dynamical Map Λ."""
    d = 2^Ns
    Λ = zeros(ComplexF64, d^2, d^2) 
    for i_s=1:d, j_s=1:d
        for i_a=1:d, j_a=1:d  
            Λ[(i_s-1)*d + j_s, (i_a-1)*d + j_a] = 
                conj(d*ρ[(i_s-1)*d + i_a, (j_s-1)*d + j_a])
        end
    end
    return Λ
end

function Λ_to_ρ(Λ, Ns)
    """
    Reverse operation of ρ_to_Λ. 
    Constructs the Choi matrix ρ of a map Λ.
    """
    d = 2^Ns
    ρ = zeros(ComplexF64, d^2, d^2)
    # The inverse relation derived from ρ_to_Λ
    for i_s=1:d, j_s=1:d
        for i_a=1:d, j_a=1:d
            row_L = (i_s-1)*d + j_s
            col_L = (i_a-1)*d + j_a
            
            row_R = (i_s-1)*d + i_a
            col_R = (j_s-1)*d + j_a
            
            ρ[row_R, col_R] = conj(Λ[row_L, col_L]) / d
        end
    end
    return ρ
end

function calculate_ρ_using_G(corr_full, qS, qA)
    Ns = 1 
    idxs = [qA, qS]
    G_sub = transpose(corr_full[idxs, idxs])
    
    # --- ROBUSTNESS FIX ---
    # We perform the calculation in the eigenbasis to safely handle 0 and 1 eigenvalues.
    F = eigen(G_sub)
    vals = real(F.values)
    vecs = F.vectors
    
    # 1. Clamp eigenvalues to avoid singularities in log() and inv()
    # Pure states have evals of 0 and 1. We shift them by epsilon.
    epsilon = 1e-9 
    vals_clamped = clamp.(vals, epsilon, 1.0 - epsilon)
    
    # 2. Reconstruct the exponent matrix α = log( G * (1-G)^-1 )
    # In the eigenbasis, this is just diagonal entries: log( λ / (1-λ) )
    temp_diag = log.(Complex.(vals_clamped ./ (1.0 .- vals_clamped)))
    α = vecs * Diagonal(temp_diag) * vecs'

    # 3. Calculate the determinant prefactor det(1-G) using clamped values
    # If we don't use clamped values, this is exactly 0, creating a 0 * Inf = NaN situation later.
    det_factor = prod(1.0 .- vals_clamped)
    # --- END FIX ---

    A = complex(zeros(2^(2*Ns), 2^(2*Ns)))
    cdag_mat, c_mat = matrix_operators(2*Ns)
    _, Sp, Sm, _ = spin_operators(2*Ns)
    
    for (i, creator_i) in enumerate(cdag_mat)
        for (j, annihilator_j) in enumerate(c_mat)
            corr_op = Matrix(creator_i) * Matrix(annihilator_j)
            A += α[i, j] * corr_op
        end
    end

    ρ = det_factor * exp(A)

    # PH transform for the Ancilla (local index 1)
    PH_gate = Sp[1] + Sm[1]
    ρ = PH_gate * ρ * PH_gate'
    
    Λ = ρ_to_Λ(ρ, Ns) 
    return Λ
end

function calculate_RHP(P; plotting=false)
    (;spec_fun, Γ_L, Γ_R, D_L, D_R, dt, N_L, β_R, μ_R) = P
    
    qA = 2*N_L + 1 
    qS = 2*N_L + 2 

    J_L = spectral_function(spec_fun, D_L, Γ_L)
    J_R = spectral_function(spec_fun, D_R, Γ_R)
    
    H = H_tot(J_L, J_R, P)
    _, _, C0_CJ = prepare_corrs(P)
    Cs_CJ = evolve_corrs(C0_CJ, H, P)
    
    times = [i*dt for i in 0:length(Cs_CJ)-1]
    rhp_accum = Float64[]
    push!(rhp_accum, 0.0)
    current_rhp = 0.0

    # --- STEP 1: Determine Steady State rho_ss ---
    Λ_final = calculate_ρ_using_G(Cs_CJ[end], qS, qA)
    
    # Project a generic state to get steady state
    d_sys = 2
    ρ_flat = vec(Matrix{ComplexF64}(I, d_sys, d_sys) ./ d_sys)
    ρ_ss = Λ_final * ρ_flat
    
    # --- CONVERGENCE SETTINGS ---
    propagator_tol = 1e-6        # Tolerance for ||L[rho_ss]||
    
    # Persistence Check:
    # Condition must hold for this many consecutive steps to count as "Converged"
    steps_required = Int(5.0 / dt)  # Require ~5.0 time units of stability
    consecutive_success = 0
    
    # Hard guard: Do not stop before this time (allows initial spikes to pass)
    min_check_time = 20.0 

    rhp_final = 0.0
    converge_time = 0.0
    for i in 1:(length(Cs_CJ)-1)
        
        # 1. Extract Map at time t
        Λ_t = calculate_ρ_using_G(Cs_CJ[i], qS, qA)
        Λ_next = calculate_ρ_using_G(Cs_CJ[i+1], qS, qA)

        # 2. Compute Intermediate Map V(t)
        F = svd(Λ_t)
        inv_S = [s > 1e-12 ? 1.0/s : 0.0 for s in F.S]
        Λ_inv = F.V * Diagonal(inv_S) * F.U'
        
        V_intermediate = Λ_next * Λ_inv
        
        # --- STEP 3: Check Propagator Memory Time (With Persistence) ---
        # Action L[rho] ≈ (V[rho] - rho) / dt
        action_on_ss = (V_intermediate * ρ_ss) - ρ_ss
        generator_norm = norm(action_on_ss) / dt
        
        # Check condition
        if generator_norm < propagator_tol && times[i] > min_check_time
            consecutive_success += 1
        else
            consecutive_success = 0 # Reset if it spikes up again
        end
    
        # -------------------------------------------------------------
        
        # 4. Regular RHP Calculation
        ρ_intermediate = Λ_to_ρ(V_intermediate, 1)
        ρ_intermediate = Hermitian(0.5 * (ρ_intermediate + ρ_intermediate'))
        
        trace_norm = sum(abs.(eigen(ρ_intermediate).values))
        diff = trace_norm - 1.0
        
        if diff > 1e-12
            current_rhp += diff 
        end
        push!(rhp_accum, current_rhp)

        #Take final value after convergence
        if consecutive_success == steps_required
            println("Steady state confirmed at t=$(times[i]) (Stable for $(steps_required*dt)). Stopping.")
            rhp_final = rhp_current / (rhp_current + 1.0)
            converge_time = i*dt 
        end
    end
    rhp_accum .= rhp_accum #./ (1.0 .+ rhp_accum)
    if plotting==true
        p = plot(times, rhp_accum, label="", xlabel="time", ylabel="Cumulative RHP Measure", lw=2, dpi=400)
        vline!(p, [converge_time], label="\$\\mathcal{L}\$ convergence time")
        display(p)
    end
    println("Final RHP Measure: ", rhp_final, ", β=$β_R μ=$μ_R")
    
    return rhp_final 
end


function calculate_BLP(P; plotting=false)
    (;spec_fun, Γ_L, Γ_R, β_R, μ_R, D_L, D_R, dt, N_L) = P
    
    qS = 2*N_L + 2 

    J_L = spectral_function(spec_fun, D_L, Γ_L)
    J_R = spectral_function(spec_fun, D_R, Γ_R)
    
    H = H_tot(J_L, J_R, P)
    C0_e, C0_f, _ = prepare_corrs(P)
    Cs_e = evolve_corrs(C0_e, H, P)
    Cs_f = evolve_corrs(C0_f, H, P)

    rho1 = [real(C[qS,qS]) for C in Cs_f]
    rho2 = [real(C[qS,qS]) for C in Cs_e]
    l1 = length(rho1)
    l2 = length(rho2)

    if l1 < l2
        rho1 = vcat(rho1, rho1[end].*ones(l2-l1))
    elseif l1 > l2
        rho2 = vcat(rho2, rho2[end].*ones(l1-l2))
    end
    
    TD = 0.5 .* abs.(rho1 .- rho2) #single-particle trace distance
    sigma = TD[2:end] .- TD[1:end-1] #change in trace distance
    
    times = dt .* collect(0:length(rho1)-2)
    blp_accum = Float64[]
    for i in 1:length(rho1)-1
        sigma_current = sigma[1:i-1]
        blp_current = sum(sigma_current[sigma_current .>= 0.0]) #sum over all positive change in trace distance
        push!(blp_accum, blp_current)
    end
    blp_final = blp_accum[end]

    if plotting==true
        p = plot(times, blp_accum, label="", xlabel="Time", ylabel="BLP Measure",
         title="\$\\Gamma=$Γ_R, \\beta=$β_R, \\mu=$μ_R\$", lw=3, dpi=400)
        display(p)
    end

    return blp_final
end

#test function
function calculate_measures(P)
    (;spec_fun, Γ_L, Γ_R, D_L, D_R, dt, N_L) = P
    
    qA = 2*N_L + 1 
    qS = 2*N_L + 2 

    println("Calculating Spectral Functions...")
    J_L = spectral_function(spec_fun, D_L, Γ_L)
    J_R = spectral_function(spec_fun, D_R, Γ_R)
    
    println("Constructing Hamiltonian and Evolving State...")
    H = H_tot(J_L, J_R, P)
    _, _, C0_CJ = prepare_corrs(P)
    Cs_CJ = evolve_corrs(C0_CJ, H, P)
    
    times = [i*dt for i in 0:length(Cs_CJ)-1]
    rhp_accum = Float64[]
    push!(rhp_accum, 0.0)
    current_rhp = 0.0

    # --- EIGENVALUE CONVERGENCE SETTINGS ---
    eval_tol = 1e-6               
    steps_required = Int(5.0 / dt)  
    consecutive_success = 0
    min_check_time = 10.0         
    
    singularity_limit = 1e6      
    
    IdentityMap = Matrix{ComplexF64}(I, 4, 4)
    evals_prev = zeros(ComplexF64, 4)
    evals_history = [Float64[] for _ in 1:4]
    
    # Trackers for diagnostic plotting
    convergence_time = NaN
    singularity_time = NaN
    
    println("Extracting RHP Measure and Generator Eigenvalues...")
    
    @showprogress for i in 1:(length(Cs_CJ)-1)
        
        Λ_t = calculate_ρ_using_G(Cs_CJ[i], qS, qA)
        Λ_next = calculate_ρ_using_G(Cs_CJ[i+1], qS, qA)

        F = svd(Λ_t)
        
        # --- THRESHOLD 1: Singularity Guard ---
        cond_number = first(F.S) / (last(F.S) + 1e-16)
        if cond_number > singularity_limit && isnan(singularity_time)
            singularity_time = times[i]
            println("Singularity limit reached at t=$(singularity_time).")
            # Continuing execution to reveal the artifact in the plot
        end
        # --------------------------------------

        # Robust SVD Inversion
        inv_S = [s > 1e-12 ? 1.0/s : 0.0 for s in F.S]
        Λ_inv = F.V * Diagonal(inv_S) * F.U'
        V_intermediate = Λ_next * Λ_inv
        
        # --- THRESHOLD 2: Eigenvalue Convergence ---
        L_t = (V_intermediate - IdentityMap) / dt
        
        evals_current = eigen(L_t).values
        sort!(evals_current, by = x -> (real(x), imag(x)))
        
        if i > 1
            eval_diff = norm(evals_current - evals_prev)
            
            if eval_diff < eval_tol && times[i] > min_check_time
                consecutive_success += 1
            else
                consecutive_success = 0 
            end
            
            if consecutive_success > steps_required && isnan(convergence_time)
                convergence_time = times[i]
                println("Generator eigenvalues converged at t=$(convergence_time) (Δλ = $eval_diff).")
                # Continuing execution to verify behavior past convergence
            end
        end
        
        for j in 1:4
            push!(evals_history[j], real(evals_current[j]))
        end
        evals_prev = evals_current
        # -------------------------------------------
        
        # Regular RHP Calculation
        ρ_intermediate = Λ_to_ρ(V_intermediate, 1)
        ρ_intermediate = Hermitian(0.5 * (ρ_intermediate + ρ_intermediate'))
        
        trace_norm = sum(abs.(eigen(ρ_intermediate).values))
        diff = trace_norm - 1.0
        
        if diff > 1e-11
            current_rhp += diff 
        end
        push!(rhp_accum, current_rhp)
    end
    
    println("Final RHP Measure: ", current_rhp)
    
    # --- PLOTTING ---
    time_axis = times[1:length(rhp_accum)]
    
    p1 = plot(time_axis, rhp_accum, 
         title="RHP Measure", 
         xlabel="Time", ylabel="RHP(t)", legend=:bottomright, linewidth=2)
         
    if !isnan(convergence_time)
        vline!(p1, [convergence_time], color=:black, linestyle=:dash, label="Convergence")
    end
    if !isnan(singularity_time)
        vline!(p1, [singularity_time], color=:red, linestyle=:dot, label="Singularity Limit")
    end
         
    time_axis_evals = times[1:length(evals_history[1])]
    
    p2 = plot(title="Real Part of Generator Eigenvalues", 
              xlabel="Time", ylabel="Re(λ_i)", legend=:bottomright)
    
    for j in 1:4
        plot!(p2, time_axis_evals, evals_history[j], label="λ_$j", linewidth=2)
    end
    
    if !isnan(convergence_time)
        vline!(p2, [convergence_time], color=:black, linestyle=:dash, label="")
    end
    if !isnan(singularity_time)
        vline!(p2, [singularity_time], color=:red, linestyle=:dot, label="")
    end
    
    p_combined = plot(p1, p2, layout=(2, 1), size=(800, 600))
    display(p_combined)
    
    return current_rhp, rhp_accum
end

end